import { Asset, Color, GSplatData, GSplatResource, Mat4, path, Quat, Texture, Vec3, Vec4 } from 'playcanvas';

import { BackendClient } from './backend';
import { BlockingPlane } from './blocking-plane';
import { BoxShape } from './box-shape';
import { EditHistory } from './edit-history';
import { SelectAllOp, SelectNoneOp, SelectInvertOp, SelectOp, StateOp, BitOp, HideSelectionOp, UnhideAllOp, DeleteSelectionOp, PagedDeleteOp, ResetOp, MultiOp, AddSplatOp, MergeOp, SplatSubdivideOp } from './edit-ops';
import { Element, ElementType } from './element';
import { Events } from './events';
import { IndexRanges } from './index-ranges';
import type { GridPlane } from './infinite-grid';
import { Scene } from './scene';
import type { ResolvedMaskSelection, ResolveMaskOptions, SelectionMaskOperation } from './selection-mask';
import { SphereShape } from './sphere-shape';
import { Splat } from './splat';
import { SingleSplat } from './splat-serialize';
import { State } from './splat-state';
import {
    DECAL_SUBDIVISION_POINT_BUDGET,
    applySubdivisionGroups,
    buildSubdivisionGroupChanges,
    planSplatSubdivision,
    subdivideSplatData
} from './splat-subdivide';
import { localize } from './ui/localization';
import { loadViewPrefs, saveViewPrefs, collectViewPrefs, applyViewPrefs } from './view-prefs';

const removeExtension = (filename: string) => {
    return filename.substring(0, filename.length - path.getExtension(filename).length);
};

// register for editor and scene events
const registerEditorEvents = (events: Events, editHistory: EditHistory, scene: Scene) => {
    const vec = new Vec3();
    const vec2 = new Vec3();
    const vec4 = new Vec4();
    const mat = new Mat4();
    const SH_C0 = 0.28209479177387814;

    const decodeColorChannel = (value: number) => {
        return Math.min(1, Math.max(0, 0.5 + value * SH_C0));
    };

    // Helper function to check if a point is blocked by any blocking plane
    const isPointBlocked = (point: Vec3, cameraPos: Vec3): boolean => {
        const blockingPlanes = events.invoke('blockingPlanes.get') as BlockingPlane[];
        if (!blockingPlanes || blockingPlanes.length === 0) {
            return false;
        }

        // Create ray from camera to point
        const direction = new Vec3().sub2(point, cameraPos).normalize();
        const distanceToPoint = point.distance(cameraPos);

        for (const plane of blockingPlanes) {
            const planePos = plane.getPlanePosition();
            const planeNormal = plane.getPlaneNormal();

            // Ray-plane intersection
            const denominator = planeNormal.dot(direction);
            if (Math.abs(denominator) > 1e-6) {
                const t = planeNormal.dot(new Vec3().sub2(planePos, cameraPos)) / denominator;
                // Check if intersection is between camera and point
                if (t > 0 && t < distanceToPoint) {
                    // Check if intersection point is within plane bounds
                    const intersectionPoint = new Vec3().add2(cameraPos, direction.clone().mulScalar(t));

                    // Transform to plane's local space
                    const planeTransform = plane.pivot.getWorldTransform();
                    const invMatrix = new Mat4().copy(planeTransform).invert();
                    const localIntersection = invMatrix.transformPoint(intersectionPoint);

                    // Check if within plane bounds (local x and z)
                    // After inverse world transform, coords are relative to original 1x1 plane geometry
                    if (Math.abs(localIntersection.x) <= 0.5 && Math.abs(localIntersection.z) <= 0.5) {
                        return true;
                    }
                }
            }
        }

        return false;
    };

    // get the list of selected splats (currently limited to just a single one)
    const selectedSplats = () => {
        const selected = events.invoke('splatSelection');
        return (selected instanceof Splat && selected.visible) ? [selected] : [];
    };

    let lastExportCursor = 0;
    let pagedDeleteSerial = 0;

    // Add unsaved changes warning (browser-only path).
    // In Electron this is handled by main.cjs via a native dialog instead, so
    // we skip the beforeunload interception there — otherwise Electron would
    // block the window close with no UI to recover, which is the root cause
    // of the "cannot quit after editing" bug.
    const isElectron = !!(window as any).electronAPI?.isElectron;
    if (!isElectron) {
        window.addEventListener('beforeunload', (e) => {
            if (!events.invoke('scene.dirty')) {
                // if the undo cursor matches last export, then we have no unsaved changes
                return undefined;
            }

            const msg = 'You have unsaved changes. Are you sure you want to leave?';
            e.returnValue = msg;
            return msg;
        });
    }

    // Register close-time save handlers for Electron. main.cjs invokes these
    // through preload's IPC channels when the user closes the window.
    if (isElectron) {
        const electronAPI = (window as any).electronAPI;
        electronAPI.registerDirtyChecker(() => {
            return {
                dirty: !!events.invoke('scene.dirty'),
                docName: events.invoke('doc.name') as string | null
            };
        });
        electronAPI.registerSaveHandler(async () => {
            try {
                await events.invoke('doc.save');
                return true;
            } catch (err) {
                console.error('[electron] doc.save failed during close:', err);
                return false;
            }
        });
        electronAPI.registerSavePromptHandler(async (docName: string) => {
            const name = docName || '未命名工程';
            const result = await events.invoke('showPopup', {
                type: 'yesno',
                header: '未保存的更改',
                message: `是否将更改保存到 "${name}"？\n关闭前保存可避免丢失未保存的修改。`
            });
            if (result.action === 'yes') return 'save';
            if (result.action === 'no') return 'discard';
            return 'cancel';
        });
    }

    events.function('targetSize', () => {
        return scene.targetSize;
    });

    events.on('scene.clear', () => {
        scene.clear();
        editHistory.clear();
        lastExportCursor = 0;
    });

    // When a splat is removed from the scene, remove all edit operations that reference it
    events.on('scene.elementRemoved', (element: Element) => {
        if (element.type === ElementType.splat) {
            editHistory.removeForSplat(element as Splat);
        }
    });

    events.function('scene.dirty', () => {
        const paintLayersDirty = events.functions.has('paint.layers.dirty') && events.invoke('paint.layers.dirty');
        return editHistory.cursor !== lastExportCursor || paintLayersDirty;
    });

    events.on('doc.saved', () => {
        lastExportCursor = editHistory.cursor;
    });

    // force render on some events

    [
        'camera.mode', 'camera.overlay', 'camera.splatSize', 'camera.fov', 'view.outlineSelection',
        'view.centersUseGaussianColor', 'view.bands', 'camera.bound', 'camera.boundDimensions', 'camera.showPoses',
        'grid.plane', 'selection.changed', 'tool.coordSpace', 'pointCloudGroup.activeGroup', 'mode.changed'
    ].forEach((eventName) => {
        events.on(eventName, () => {
            scene.forceRender = true;
        });
    });

    // grid.visible

    const setGridVisible = (visible: boolean) => {
        if (visible !== scene.grid.visible) {
            scene.grid.visible = visible;
            events.fire('grid.visible', visible);
        }
    };

    events.function('grid.visible', () => {
        return scene.grid.visible;
    });

    events.on('grid.setVisible', (visible: boolean) => {
        setGridVisible(visible);
    });

    let gizmoKeyHidden = false;
    let savedSelectedAlpha: number | null = null;
    let savedBoundVisible: boolean | null = null;

    const setGizmoKeyHidden = (hidden: boolean) => {
        if (hidden === gizmoKeyHidden) return;
        gizmoKeyHidden = hidden;

        if (hidden) {
            const clr = events.invoke('selectedClr') as Color;
            savedSelectedAlpha = clr.a;
            events.fire('setSelectedClr', new Color(clr.r, clr.g, clr.b, 0));
            savedBoundVisible = events.invoke('camera.bound') as boolean;
            events.fire('camera.setBound', false);
            events.fire('gizmo.keyHide');
        } else {
            if (savedSelectedAlpha !== null) {
                const clr = events.invoke('selectedClr') as Color;
                events.fire('setSelectedClr', new Color(clr.r, clr.g, clr.b, savedSelectedAlpha));
                savedSelectedAlpha = null;
            }
            if (savedBoundVisible !== null) {
                events.fire('camera.setBound', savedBoundVisible);
                savedBoundVisible = null;
            }
            events.fire('gizmo.keyShow');
        }
    };

    events.function('gizmo.keyHidden', () => gizmoKeyHidden);

    events.on('gizmo.setKeyHidden', (hidden: boolean) => {
        setGizmoKeyHidden(hidden);
    });

    events.on('grid.toggleVisible', () => {
        setGridVisible(!scene.grid.visible);
        setGizmoKeyHidden(!gizmoKeyHidden);
    });

    setGridVisible(scene.config.show.grid);

    // grid.plane

    const setGridPlane = (plane: GridPlane) => {
        if (plane !== scene.grid.plane) {
            scene.grid.plane = plane;
            events.fire('grid.plane', plane);
        }
    };

    events.function('grid.plane', () => scene.grid.plane);

    events.on('grid.setPlane', (plane: GridPlane) => {
        setGridPlane(plane);
    });

    // camera.fovDolly

    let fovDolly = false;

    const setFovDolly = (value: boolean) => {
        if (value !== fovDolly) {
            fovDolly = value;
            events.fire('camera.fovDolly', fovDolly);
        }
    };

    events.function('camera.fovDolly', () => fovDolly);

    events.on('camera.setFovDolly', (value: boolean) => {
        setFovDolly(value);
    });

    // camera.fov

    const setCameraFov = (fov: number) => {
        const { camera } = scene;
        if (fov !== camera.fov) {
            const oldFovFactor = camera.fovFactor;
            camera.fov = fov;

            // Normal FOV changes behave like a lens zoom and keep the camera
            // position fixed. Auto-dolly instead preserves apparent framing.
            if (!fovDolly) {
                const { controls } = scene.config;
                const scale = camera.fovFactor / oldFovFactor;
                const tween = camera.distanceTween;
                for (const state of [tween.value, tween.source, tween.target]) {
                    state.distance = Math.max(controls.minZoom, Math.min(controls.maxZoom, state.distance * scale));
                }
            }

            events.fire('camera.fov', camera.fov);
        }
    };

    events.function('camera.fov', () => {
        return scene.camera.fov;
    });

    events.on('camera.setFov', (fov: number) => {
        setCameraFov(fov);
    });

    // camera.tonemapping

    events.function('camera.tonemapping', () => {
        return scene.camera.tonemapping;
    });

    events.on('camera.setTonemapping', (value: string) => {
        scene.camera.tonemapping = value;
    });

    // camera.bound

    let bound = scene.config.show.bound;

    const setBoundVisible = (visible: boolean) => {
        if (visible !== bound) {
            bound = visible;
            events.fire('camera.bound', bound);
        }
    };

    events.function('camera.bound', () => {
        return bound;
    });

    events.on('camera.setBound', (value: boolean) => {
        setBoundVisible(value);
    });

    events.on('camera.toggleBound', () => {
        setBoundVisible(!events.invoke('camera.bound'));
    });

    // camera.boundDimensions

    let boundDimensions = scene.config.show.boundDimensions;

    const setBoundDimensionsVisible = (visible: boolean) => {
        if (visible !== boundDimensions) {
            boundDimensions = visible;
            events.fire('camera.boundDimensions', boundDimensions);
        }
    };

    events.function('camera.boundDimensions', () => {
        return boundDimensions;
    });

    events.on('camera.setBoundDimensions', (value: boolean) => {
        setBoundDimensionsVisible(value);
    });

    events.on('camera.toggleBoundDimensions', () => {
        setBoundDimensionsVisible(!events.invoke('camera.boundDimensions'));
    });

    // camera.showPoses

    let showPoses = scene.config.show.cameraPoses;

    const setShowPoses = (visible: boolean) => {
        if (visible !== showPoses) {
            showPoses = visible;
            events.fire('camera.showPoses', showPoses);
        }
    };

    events.function('camera.showPoses', () => {
        return showPoses;
    });

    events.on('camera.setShowPoses', (value: boolean) => {
        setShowPoses(value);
    });

    events.on('camera.toggleShowPoses', () => {
        setShowPoses(!events.invoke('camera.showPoses'));
    });

    // camera.focus

    events.on('camera.focus', () => {
        const splat = selectedSplats()[0];
        const shapeSel = events.invoke('shapeSelection') as Element | null;

        if (shapeSel instanceof BoxShape || shapeSel instanceof SphereShape) {
            // Focus on wrapper shape
            const pivot = shapeSel.pivot;
            const focalPoint = pivot.getPosition();
            const radius = shapeSel instanceof SphereShape ?
                shapeSel.radius :
                pivot.getLocalScale().length() / 2;
            scene.camera.focus({
                focalPoint: focalPoint.clone(),
                radius,
                speed: 1
            });
        } else if (splat) {
            // use current bounds (caller should have awaited the operation that changed data)
            const bound = splat.numSelected > 0 ?
                splat.selectionBound :
                splat.localBound;
            vec.copy(bound.center);

            const worldTransform = splat.worldTransform;
            worldTransform.transformPoint(vec, vec);
            worldTransform.getScale(vec2);

            scene.camera.focus({
                focalPoint: vec,
                radius: bound.halfExtents.length() * vec2.x,
                speed: 1
            });
        } else {
            // Nothing selected — reset camera to initial position
            const { initialAzim, initialElev, initialZoom } = scene.config.controls;
            const x = Math.sin(initialAzim * Math.PI / 180) * Math.cos(initialElev * Math.PI / 180);
            const y = -Math.sin(initialElev * Math.PI / 180);
            const z = Math.cos(initialAzim * Math.PI / 180) * Math.cos(initialElev * Math.PI / 180);
            const zoom = initialZoom;
            scene.camera.setPose(new Vec3(x * zoom, y * zoom, z * zoom), new Vec3(0, 0, 0));
        }
    });

    events.on('camera.reset', () => {
        const { initialAzim, initialElev, initialZoom } = scene.config.controls;
        const x = Math.sin(initialAzim * Math.PI / 180) * Math.cos(initialElev * Math.PI / 180);
        const y = -Math.sin(initialElev * Math.PI / 180);
        const z = Math.cos(initialAzim * Math.PI / 180) * Math.cos(initialElev * Math.PI / 180);
        const zoom = initialZoom;

        scene.camera.setPose(new Vec3(x * zoom, y * zoom, z * zoom), new Vec3(0, 0, 0));
    });

    // handle camera align events
    events.on('camera.align', (axis: string) => {
        switch (axis) {
            case 'px': scene.camera.setAzimElev(90, 0); break;
            case 'py': scene.camera.setAzimElev(0, -90); break;
            case 'pz': scene.camera.setAzimElev(0, 0); break;
            case 'nx': scene.camera.setAzimElev(270, 0); break;
            case 'ny': scene.camera.setAzimElev(0, 90); break;
            case 'nz': scene.camera.setAzimElev(180, 0); break;
        }

        // switch to ortho mode
        scene.camera.ortho = true;
    });

    // returns true if the selected splat has selected gaussians
    events.function('selection.splats', () => {
        const splat = events.invoke('splatSelection') as Splat;
        return splat?.numSelected > 0;
    });

    // returns true if any splat is selected (regardless of gaussian selection)
    events.function('selection.hasSplat', () => {
        const splat = events.invoke('splatSelection') as Splat;
        return splat instanceof Splat;
    });

    events.on('select.all', () => {
        selectedSplats().forEach((splat) => {
            splat.lodEditLog?.onEditHistoryAdd();
            splat.lodEditLog?.recordSelectAll(splat);
            events.fire('edit.add', new SelectAllOp(splat));
        });
    });

    events.on('select.none', () => {
        const splats = scene.getElementsByType(ElementType.splat) as Splat[];
        splats.forEach((splat) => {
            splat.lodEditLog?.onEditHistoryAdd();
            splat.lodEditLog?.recordSelectNone(splat);
            events.fire('edit.add', new SelectNoneOp(splat));
        });
    });

    events.on('select.invert', () => {
        selectedSplats().forEach((splat) => {
            splat.lodEditLog?.onEditHistoryAdd();
            splat.lodEditLog?.recordSelectInvert(splat);
            events.fire('edit.add', new SelectInvertOp(splat));
        });
    });

    events.on('select.mask', (op: 'add'|'remove'|'set', mask: Uint8Array | Uint32Array) => {
        selectedSplats().forEach((splat) => {
            fireSelectWithLog(splat, op, mask);
        });
    });

    // Wrapper that records the selection to the splat's LodEditLog (if present)
    // BEFORE constructing SelectOp, because SelectOp consumes sel in its
    // constructor. For non-LCC splats lodEditLog is null and this is a no-op.
    const fireSelectWithLog = (splat: Splat, op: SelectionMaskOperation, sel: Uint8Array | Uint32Array) => {
        // 独立编辑激活时，剔除组外点云，使选择/选区工具只能命中组内点云
        const edited = events.invoke('pointCloudGroup.editActive') ? splat.desaturateMaskData : null;
        let filteredSel = sel;
        if (edited && edited.length > 0) {
            if (op === 'set') {
                // A set operation must preserve selection outside the active
                // independent-edit group. Converting to a mask lets us carry
                // the existing state for protected rows rather than treating
                // them as misses (which would incorrectly deselect them).
                const state = splat.splatData.getProp('state') as Uint8Array;
                const mask = sel instanceof Uint32Array ? new Uint8Array(state.length) : new Uint8Array(sel);
                if (sel instanceof Uint32Array) {
                    for (let i = 0; i < sel.length; i++) {
                        if (sel[i] < mask.length) mask[sel[i]] = 255;
                    }
                }
                const n = Math.min(mask.length, edited.length);
                for (let i = 0; i < n; i++) {
                    if (edited[i] !== 0) mask[i] = (state[i] & State.selected) !== 0 ? 255 : 0;
                }
                filteredSel = mask;
            } else if (sel instanceof Uint32Array) {
                const keep: number[] = [];
                for (let i = 0; i < sel.length; i++) {
                    const idx = sel[i];
                    if (idx < edited.length && edited[idx] === 0) keep.push(idx);
                }
                filteredSel = new Uint32Array(keep);
            } else {
                const n = Math.min(sel.length, edited.length);
                for (let i = 0; i < n; i++) {
                    if (edited[i] !== 0) sel[i] = 0;
                }
            }
        }
        splat.lodEditLog?.onEditHistoryAdd();
        splat.lodEditLog?.recordSelect(splat, op, filteredSel);
        events.fire('edit.add', new SelectOp(splat, op, filteredSel));
        // edit.add queues the async GPU state update synchronously. Append a
        // barrier so callers that need a committed selection can wait for it
        // without bypassing the edit.add observers used by the UI.
        return scene.commandQueue.enqueue(() => {});
    };

    const filterBlockingPlanes = (splat: Splat, data: Uint8Array) => {
        const blockingPlanes = events.invoke('blockingPlanes.get') as BlockingPlane[];
        if (!blockingPlanes || blockingPlanes.length === 0) return;

        const splatData = splat.splatData;
        const x = splatData.getProp('x') as Float32Array;
        const y = splatData.getProp('y') as Float32Array;
        const z = splatData.getProp('z') as Float32Array;
        if (!x || !y || !z) return;

        const cameraPos = scene.camera.position;
        const worldPos = new Vec3();
        for (let i = 0; i < data.length; i++) {
            if (data[i] !== 255) continue;
            worldPos.set(x[i], y[i], z[i]);
            splat.worldTransform.transformPoint(worldPos, worldPos);
            if (isPointBlocked(worldPos, cameraPos)) {
                data[i] = 0;
            }
        }
    };

    const resolveCenters = (splat: Splat, options: any, boundOptions?: any): Promise<Uint8Array> => {
        // run the GPU intersect inside one queued task so the gpu readback is
        // ordered relative to other queued history ops (rapid drag + undo,
        // drag-while-camera-settling, etc).
        return scene.commandQueue.enqueue(async () => {
            const data = await scene.dataProcessor.intersect(options, splat);
            try {
                // The packed GPU mask is texture-aligned and can contain tail
                // bytes beyond the actual Gaussian count. Public selection
                // snapshots must always be exactly one byte per Gaussian so
                // visible-ID observations can be combined with center hits.
                const hits = data.subarray(0, splat.splatData.numSplats);
                if (boundOptions) {
                    const boundData = await scene.dataProcessor.intersect(boundOptions, splat);
                    try {
                        for (let i = 0; i < hits.length; i++) {
                            hits[i] = hits[i] && boundData[i] ? 255 : 0;
                        }
                    } finally {
                        scene.dataProcessor.releaseMask(boundData);
                    }
                }
                filterBlockingPlanes(splat, hits);
                // GPU masks come from a pool. Commit a private snapshot before
                // returning so later queued work cannot overwrite this result.
                return new Uint8Array(hits);
            } finally {
                scene.dataProcessor.releaseMask(data);
            }
        });
    };

    const intersectCenters = async (splat: Splat, op: SelectionMaskOperation, options: any) => {
        const hits = await resolveCenters(splat, options);
        fireSelectWithLog(splat, op, hits);
    };

    // Helper: get GPU intersect options for a selected bound shape (BoxShape or SphereShape)
    const getBoundIntersectOptions = (selection: BoxShape | SphereShape) => {
        if (selection instanceof BoxShape) {
            const p = selection.pivot.getPosition();
            const r = selection.pivot.getLocalRotation();
            return {
                box: { x: p.x, y: p.y, z: p.z, lenx: selection.lenX, leny: selection.lenY, lenz: selection.lenZ, rx: r.x, ry: r.y, rz: r.z, rw: r.w }
            };
        }
        const p = selection.pivot.getPosition();
        const r = selection.pivot.getLocalRotation();
        return {
            sphere: { x: p.x, y: p.y, z: p.z, radiusX: selection.radiusX, radiusY: selection.radiusY, radiusZ: selection.radiusZ, rx: r.x, ry: r.y, rz: r.z, rw: r.w }
        };

    };

    // Helper: CPU-side containment check for a point against a bound shape
    const isInsideBoundShape = (worldPoint: Vec3, shape: BoxShape | SphereShape, invRot: Quat, shapePos: Vec3) => {
        const local = new Vec3();
        local.sub2(worldPoint, shapePos);
        invRot.transformVector(local, local);
        if (shape instanceof BoxShape) {
            return Math.abs(local.x) <= shape.lenX / 2 &&
                Math.abs(local.y) <= shape.lenY / 2 &&
                Math.abs(local.z) <= shape.lenZ / 2;
        }
        return (local.x * local.x) / (shape.radiusX * shape.radiusX) +
             (local.y * local.y) / (shape.radiusY * shape.radiusY) +
             (local.z * local.z) / (shape.radiusZ * shape.radiusZ) <= 1;
    };

    events.on('select.bySphere', async (op: 'add'|'remove'|'set', sphere: number[]) => {
        const allSplats = scene.getElementsByType(ElementType.splat) as Splat[];
        for (const splat of allSplats) {
            if (splat.visible) {
                await intersectCenters(splat, op, {
                    sphere: { x: sphere[0], y: sphere[1], z: sphere[2], radiusX: sphere[3], radiusY: sphere[4], radiusZ: sphere[5], rx: sphere[6], ry: sphere[7], rz: sphere[8], rw: sphere[9] }
                });
            }
        }
    });

    events.on('select.byBox', async (op: 'add'|'remove'|'set', box: number[]) => {
        const allSplats = scene.getElementsByType(ElementType.splat) as Splat[];
        for (const splat of allSplats) {
            if (splat.visible) {
                await intersectCenters(splat, op, {
                    box: { x: box[0], y: box[1], z: box[2], lenx: box[3], leny: box[4], lenz: box[5], rx: box[6], ry: box[7], rz: box[8], rw: box[9] }
                });
            }
        }
    });

    events.function('select.rect', async (op: 'add'|'remove'|'set', rect: any) => {
        const mode = events.invoke('camera.mode');
        const overlay = events.invoke('camera.overlay');
        const { width, height } = scene.targetSize;

        const shapeSel = events.invoke('shapeSelection');
        const boundOptions = (shapeSel instanceof BoxShape || shapeSel instanceof SphereShape) ?
            getBoundIntersectOptions(shapeSel) : null;
        let boundInvRot: Quat | null = null;
        let boundPos: Vec3 | null = null;
        if (shapeSel instanceof BoxShape || shapeSel instanceof SphereShape) {
            boundInvRot = new Quat();
            boundInvRot.copy(shapeSel.pivot.getLocalRotation()).invert();
            boundPos = shapeSel.pivot.getPosition().clone();
        }

        for (const splat of selectedSplats()) {
            if (mode === 'centers' || overlay) {
                const splatData = splat.splatData;
                const x = splatData.getProp('x');
                const y = splatData.getProp('y');
                const z = splatData.getProp('z');

                const camera = scene.camera.camera;
                const cameraPos = scene.camera.position;
                const worldPos = new Vec3();

                // calculate final matrix
                mat.mul2(camera.camera._viewProjMat, splat.worldTransform);

                const numSplats = splatData.numSplats;
                const mask = new Uint8Array(numSplats);

                // Convert normalized rect to pixel coordinates
                const sx1 = rect.start.x * width;
                const sy1 = rect.start.y * height;
                const sx2 = rect.end.x * width;
                const sy2 = rect.end.y * height;
                const minX = Math.min(sx1, sx2);
                const maxX = Math.max(sx1, sx2);
                const minY = Math.min(sy1, sy2);
                const maxY = Math.max(sy1, sy2);

                for (let i = 0; i < numSplats; i++) {
                    vec4.set(x[i], y[i], z[i], 1.0);
                    mat.transformVec4(vec4, vec4);
                    const px = (vec4.x / vec4.w * 0.5 + 0.5) * width;
                    const py = (-vec4.y / vec4.w * 0.5 + 0.5) * height;

                    if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
                        // Check if splat is blocked by any blocking plane
                        worldPos.set(x[i], y[i], z[i]);
                        splat.worldTransform.transformPoint(worldPos, worldPos);
                        if (!isPointBlocked(worldPos, cameraPos)) {
                            if (boundInvRot && boundPos && shapeSel) {
                                if (isInsideBoundShape(worldPos, shapeSel as BoxShape | SphereShape, boundInvRot, boundPos)) {
                                    mask[i] = 255;
                                }
                            } else {
                                mask[i] = 255;
                            }
                        }
                    }
                }

                fireSelectWithLog(splat, op, mask);
            } else {
                scene.camera.pickPrep(splat, op);
                const pick = await scene.camera.pickRect(
                    rect.start.x,
                    rect.start.y,
                    rect.end.x - rect.start.x,
                    rect.end.y - rect.start.y
                );

                // Filter out blocked points
                const cameraPos = scene.camera.position;
                const worldPos = new Vec3();
                const x = splat.splatData.getProp('x');
                const y = splat.splatData.getProp('y');
                const z = splat.splatData.getProp('z');

                const filteredIds = new Set<number>();
                const uniqueIds = new Set(pick);

                for (const pickId of uniqueIds) {
                    if (pickId !== undefined && pickId !== 0xffffffff && x && y && z && pickId < x.length) {
                        worldPos.set(x[pickId], y[pickId], z[pickId]);
                        splat.worldTransform.transformPoint(worldPos, worldPos);
                        if (!isPointBlocked(worldPos, cameraPos)) {
                            filteredIds.add(pickId);
                        }
                    }
                }

                // If a bound shape is selected, filter to only points inside the shape
                if (boundOptions) {
                    const numSplats = splat.splatData.numSplats;
                    const blockMask = new Uint8Array(numSplats);
                    for (const id of filteredIds) {
                        if (id < numSplats) blockMask[id] = 255;
                    }
                    const boundMask = await scene.dataProcessor.intersect(boundOptions, splat);
                    for (let i = 0; i < numSplats; i++) {
                        blockMask[i] = blockMask[i] && boundMask[i];
                    }
                    scene.dataProcessor.releaseMask(boundMask);
                    filteredIds.clear();
                    for (let i = 0; i < numSplats; i++) {
                        if (blockMask[i] === 255) filteredIds.add(i);
                    }
                }

                const sortedIds = new Uint32Array(filteredIds).sort();
                fireSelectWithLog(splat, op, sortedIds);
            }
        }
    });

    events.function('select.resolveMask', async (
        canvas: HTMLCanvasElement,
        context: CanvasRenderingContext2D,
        options: ResolveMaskOptions = {}
    ): Promise<ResolvedMaskSelection | null> => {
        const mode = events.invoke('camera.mode');
        const overlay = events.invoke('camera.overlay');
        const projection = options.projection ?? 'auto';

        // When a bound shape is selected, restrict to points inside the shape.
        const shapeSel = events.invoke('shapeSelection');
        const boundOptions = (shapeSel instanceof BoxShape || shapeSel instanceof SphereShape) ?
            getBoundIntersectOptions(shapeSel) : null;
        const splat = selectedSplats()[0];
        if (!splat || canvas.width === 0 || canvas.height === 0) return null;

        if (projection === 'centers' || (projection === 'auto' && (mode === 'centers' || overlay))) {
            // Snapshot the source before awaiting GPU work. Selection tools share
            // their drawing canvas and may clear or redraw it immediately.
            const snapshot = document.createElement('canvas');
            snapshot.width = canvas.width;
            snapshot.height = canvas.height;
            const snapshotContext = snapshot.getContext('2d');
            if (!snapshotContext) return null;
            snapshotContext.drawImage(canvas, 0, 0);

            const maskTexture = new Texture(scene.graphicsDevice);
            try {
                maskTexture.setSource(snapshot);
                const hits = await resolveCenters(splat, { mask: maskTexture }, boundOptions);
                return { splat, hits };
            } finally {
                maskTexture.destroy();
            }
        }

        // getImageData commits an operation-private CPU snapshot before the
        // asynchronous ID pick starts.
        const mask = context.getImageData(0, 0, canvas.width, canvas.height);

        let mx0 = mask.width;
        let my0 = mask.height;
        let mx1 = -1;
        let my1 = -1;
        for (let y = 0; y < mask.height; ++y) {
            for (let x = 0; x < mask.width; ++x) {
                if (mask.data[(y * mask.width + x) * 4 + 3] > 0) {
                    mx0 = Math.min(mx0, x);
                    my0 = Math.min(my0, y);
                    mx1 = Math.max(mx1, x);
                    my1 = Math.max(my1, y);
                }
            }
        }

        // An empty gesture/semantic mask is not a request to clear selection.
        if (mx1 < mx0 || my1 < my0) return null;

        const nx0 = mx0 / mask.width;
        const ny0 = my0 / mask.height;
        const nx1 = (mx1 + 1) / mask.width;
        const ny1 = (my1 + 1) / mask.height;
        const nw = nx1 - nx0;
        const nh = ny1 - ny0;

        const collectVisible = options.collectVisible === true;
        const pickX = collectVisible ? 0 : nx0;
        const pickY = collectVisible ? 0 : ny0;
        const pickWidth = collectVisible ? 1 : nw;
        const pickHeight = collectVisible ? 1 : nh;

        scene.camera.pickPrep(splat, 'set', options.alphaThreshold ?? 0);
        const pick = await scene.camera.pickRect(pickX, pickY, pickWidth, pickHeight);

        const { width, height } = scene.targetSize;
        const px = Math.floor(pickX * width);
        const py = Math.floor(pickY * height);
        const pw = Math.max(1, Math.ceil((pickX + pickWidth) * width) - px);
        const ph = Math.max(1, Math.ceil((pickY + pickHeight) * height) - py);

        const hits = new Uint8Array(splat.splatData.numSplats);
        const visible = collectVisible ? new Uint8Array(hits.length) : undefined;
        const sourceConfidence = options.sourceConfidence?.length === mask.width * mask.height ?
            options.sourceConfidence : undefined;
        const coverage = collectVisible && sourceConfidence ? new Uint16Array(hits.length) : undefined;
        const confidenceSums = coverage ? new Uint32Array(hits.length) : undefined;
        for (let y = 0; y < ph; ++y) {
            for (let x = 0; x < pw; ++x) {
                const pickId = pick[(ph - 1 - y) * pw + x];
                if (pickId !== undefined && pickId !== 0xffffffff && pickId < hits.length) {
                    if (visible) visible[pickId] = 255;
                    const mx = Math.min(mask.width - 1, Math.floor((pickX + x / width) * mask.width));
                    const my = Math.min(mask.height - 1, Math.floor((pickY + y / height) * mask.height));
                    if (coverage && confidenceSums && coverage[pickId] < 0xffff) {
                        coverage[pickId]++;
                        confidenceSums[pickId] += sourceConfidence![my * mask.width + mx];
                    }
                    if (mask.data[(my * mask.width + mx) * 4] > 127) hits[pickId] = 255;
                }
            }
        }

        filterBlockingPlanes(splat, hits);
        if (visible) filterBlockingPlanes(splat, visible);

        if (boundOptions) {
            const boundMask = await scene.commandQueue.enqueue(async () => {
                const pooled = await scene.dataProcessor.intersect(boundOptions, splat);
                try {
                    return new Uint8Array(pooled);
                } finally {
                    scene.dataProcessor.releaseMask(pooled);
                }
            });
            for (let i = 0; i < hits.length; i++) {
                hits[i] = hits[i] && boundMask[i] ? 255 : 0;
                if (visible) visible[i] = visible[i] && boundMask[i] ? 255 : 0;
            }
        }

        const confidence = coverage && confidenceSums ? new Uint8Array(hits.length) : undefined;
        if (confidence && coverage && confidenceSums) {
            for (let i = 0; i < confidence.length; i++) {
                if (coverage[i] > 0 && visible?.[i]) confidence[i] = Math.round(confidenceSums[i] / coverage[i]);
            }
        }
        return { splat, hits, visible, coverage, confidence, predictedIou: options.predictedIou };
    });

    events.function('select.applyResolvedMask', async (op: SelectionMaskOperation, resolved: ResolvedMaskSelection) => {
        if (!resolved || resolved.splat.scene !== scene) return false;
        await fireSelectWithLog(resolved.splat, op, resolved.hits);
        return true;
    });

    events.function('select.byMask', async (op: SelectionMaskOperation, canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) => {
        const resolved = await events.invoke('select.resolveMask', canvas, context) as ResolvedMaskSelection | null;
        if (resolved) {
            fireSelectWithLog(resolved.splat, op, resolved.hits);
        }
    });

    events.function('select.point', async (op: 'add'|'remove'|'set', point: { x: number, y: number }) => {
        const { width, height } = scene.targetSize;
        const mode = events.invoke('camera.mode');
        const overlay = events.invoke('camera.overlay');

        const shapeSel = events.invoke('shapeSelection');
        let boundInvRot: Quat | null = null;
        let boundPos: Vec3 | null = null;
        if (shapeSel instanceof BoxShape || shapeSel instanceof SphereShape) {
            boundInvRot = new Quat();
            boundInvRot.copy(shapeSel.pivot.getLocalRotation()).invert();
            boundPos = shapeSel.pivot.getPosition().clone();
        }

        for (const splat of selectedSplats()) {
            const splatData = splat.splatData;

            if (mode === 'centers' || overlay) {
                const x = splatData.getProp('x');
                const y = splatData.getProp('y');
                const z = splatData.getProp('z');

                const splatSize = events.invoke('camera.splatSize');
                const camera = scene.camera.camera;
                const sx = point.x * width;
                const sy = point.y * height;

                // calculate final matrix
                mat.mul2(camera.camera._viewProjMat, splat.worldTransform);

                const numSplats = splatData.numSplats;
                const mask = new Uint8Array(numSplats);
                const cameraPos = scene.camera.position;
                const worldPos = new Vec3();
                for (let i = 0; i < numSplats; i++) {
                    vec4.set(x[i], y[i], z[i], 1.0);
                    mat.transformVec4(vec4, vec4);
                    const px = (vec4.x / vec4.w * 0.5 + 0.5) * width;
                    const py = (-vec4.y / vec4.w * 0.5 + 0.5) * height;
                    if (Math.abs(px - sx) < splatSize && Math.abs(py - sy) < splatSize) {
                        // Check if splat is blocked by any blocking plane
                        worldPos.set(x[i], y[i], z[i]);
                        splat.worldTransform.transformPoint(worldPos, worldPos);
                        if (!isPointBlocked(worldPos, cameraPos)) {
                            if (boundInvRot && boundPos && shapeSel) {
                                if (isInsideBoundShape(worldPos, shapeSel as BoxShape | SphereShape, boundInvRot, boundPos)) {
                                    mask[i] = 255;
                                }
                            } else {
                                mask[i] = 255;
                            }
                        }
                    }
                }

                fireSelectWithLog(splat, op, mask);
            } else {
                scene.camera.pickPrep(splat, op);

                // Use normalized coordinates with minimal size for single pixel pick
                const pickResult = await scene.camera.pickRect(
                    point.x,
                    point.y,
                    1 / width,
                    1 / height
                );
                const pickId = pickResult[0];
                // Check if picked splat is blocked by any blocking plane
                if (pickId !== undefined && pickId !== 0xffffffff) {
                    const x = splat.splatData.getProp('x');
                    const y = splat.splatData.getProp('y');
                    const z = splat.splatData.getProp('z');
                    if (x && y && z && pickId < x.length) {
                        const worldPos = new Vec3(x[pickId], y[pickId], z[pickId]);
                        splat.worldTransform.transformPoint(worldPos, worldPos);
                        const cameraPos = scene.camera.position;
                        if (isPointBlocked(worldPos, cameraPos)) {
                            fireSelectWithLog(splat, op, new Uint32Array([]));
                            return;
                        }
                        if (boundInvRot && boundPos && shapeSel) {
                            if (!isInsideBoundShape(worldPos, shapeSel as BoxShape | SphereShape, boundInvRot, boundPos)) {
                                fireSelectWithLog(splat, op, new Uint32Array([]));
                                return;
                            }
                        }
                    }
                }
                fireSelectWithLog(splat, op, new Uint32Array([pickId]));
            }
        }
    });

    // Eyedropper selection with SelectOp so undo/redo and selection state updates remain consistent.
    // Threshold acts as a per-channel absolute difference: 0 only matches identical colors while 1 matches everything.
    // TO DO:
    // -  alternative distance metrics such as HSV.
    // -  alternative UI for threshold, two handles for min/max?
    events.function('select.colorMatch', async (op: 'add'|'remove'|'set', point: { x: number, y: number }, threshold = 0) => {
        const splats = selectedSplats();
        const targetSize = scene.targetSize;
        if (!splats.length || !targetSize || !point) {
            return;
        }

        const { width, height } = targetSize;
        if (!width || !height) {
            return;
        }

        // Clamp normalized coordinates to valid range
        const nx = Math.max(0, Math.min(1, point.x));
        const ny = Math.max(0, Math.min(1, point.y));
        const colorThreshold = Math.min(1, Math.max(0, Number.isFinite(threshold) ? threshold : 0));

        // When a bound shape is selected, only match colors within the shape
        const shapeSel = events.invoke('shapeSelection');
        const boundOptions = (shapeSel instanceof BoxShape || shapeSel instanceof SphereShape) ?
            getBoundIntersectOptions(shapeSel) : null;

        for (const splat of splats) {
            scene.camera.pickPrep(splat, 'set');
            // Use normalized coordinates with minimal size for single pixel pick
            const pickBuffer = await scene.camera.pickRect(nx, ny, 1 / width, 1 / height);
            const pickId = pickBuffer?.[0];
            if (pickId === undefined || pickId === 0xffffffff) {
                continue;
            }

            const reds = splat.splatData.getProp('f_dc_0') as Float32Array;
            const greens = splat.splatData.getProp('f_dc_1') as Float32Array;
            const blues = splat.splatData.getProp('f_dc_2') as Float32Array;
            // validate pickId and color channels exist
            if (!reds || !greens || !blues || pickId < 0 || pickId >= reds.length) {
                continue;
            }
            // decode color channels for the reference pixel
            const refR = decodeColorChannel(reds[pickId]);
            const refG = decodeColorChannel(greens[pickId]);
            const refB = decodeColorChannel(blues[pickId]);

            // materialize hits into an owned mask up front; SelectOp consumes
            // a committed snapshot.
            const numSplats = splat.splatData.numSplats;
            const mask = new Uint8Array(numSplats);
            for (let i = 0; i < numSplats; i++) {
                if (Math.abs(decodeColorChannel(reds[i]) - refR) <= colorThreshold &&
                    Math.abs(decodeColorChannel(greens[i]) - refG) <= colorThreshold &&
                    Math.abs(decodeColorChannel(blues[i]) - refB) <= colorThreshold) {
                    mask[i] = 255;
                }
            }

            // If a bound shape is selected, only keep points inside it
            if (boundOptions) {
                const boundMask = await scene.dataProcessor.intersect(boundOptions, splat);
                for (let i = 0; i < numSplats; i++) {
                    mask[i] = mask[i] && boundMask[i];
                }
                scene.dataProcessor.releaseMask(boundMask);
            }

            events.fire('edit.add', new SelectOp(splat, op, mask));
        }
    });

    events.on('select.hide', () => {
        selectedSplats().forEach((splat) => {
            events.fire('edit.add', new HideSelectionOp(splat));
        });
    });

    events.on('select.unhide', () => {
        selectedSplats().forEach((splat) => {
            events.fire('edit.add', new UnhideAllOp(splat));
        });
    });

    events.on('select.delete', async () => {
        // Don't delete gaussians when measure tool is active (backspace deletes measure points instead)
        if (events.invoke('tool.active') === 'measure') {
            return;
        }

        for (const splat of selectedSplats()) {
            const pagedSession = splat.pagedLodEditSession;
            if (!pagedSession || !splat.lodEditLog) {
                splat.lodEditLog?.onEditHistoryAdd();
                splat.lodEditLog?.recordDelete(splat);
                await editHistory.add(new DeleteSelectionOp(splat));
                continue;
            }

            const voxel = splat.lodEditLog.captureSelectedVoxel(splat);
            if (!voxel) continue;

            // Delete the proxy immediately. Exact LOD0 refinement continues in
            // the background and resolves the first operation when its source
            // rows arrive, so the user does not wait on a page/network roundtrip.
            splat.lodEditLog.onEditHistoryAdd();
            splat.lodEditLog.recordDelete(splat);
            const id = `paged-delete-${Date.now()}-${pagedDeleteSerial++}`;
            const pagedDeleteOp = new PagedDeleteOp(
                pagedSession,
                id,
                new Uint32Array(),
                voxel,
                true
            );
            try {
                await editHistory.add(new MultiOp([
                    pagedDeleteOp,
                    new DeleteSelectionOp(splat)
                ]));
            } catch (error) {
                console.error('[editor] Failed to apply proxy deletion', error);
                continue;
            }

            const refinement = pagedSession.startDeleteRefinement(
                id,
                voxel,
                sourceIndices => pagedDeleteOp.resolve(sourceIndices)
            );
            refinement.catch((error) => {
                // Keep the fast proxy edit visible. The failed task remains
                // observable in the console, while save/export will surface
                // the rejected pending task before producing output.
                console.error('[editor] Background LOD0 deletion refinement failed', error);
            });
        }
    });

    // Opacity threshold selection - selects all points with opacity below the given threshold
    // This is useful for data cleanup by selecting low-opacity points that are barely visible
    events.on('select.opacityThreshold', (op: 'add'|'remove'|'set', threshold: number) => {
        const splats = selectedSplats();
        if (!splats.length) {
            return;
        }

        const opacityThreshold = Math.min(1, Math.max(0, Number.isFinite(threshold) ? threshold : 0));

        splats.forEach((splat) => {
            const opacities = splat.splatData.getProp('opacity') as Float32Array;
            if (!opacities) {
                return;
            }

            const numSplats = splat.splatData.numSplats;
            const mask = new Uint8Array(numSplats);
            for (let i = 0; i < numSplats; i++) {
                // Convert logit to probability using sigmoid function
                const opacity = 1 / (1 + Math.exp(-opacities[i]));
                mask[i] = opacity < opacityThreshold ? 255 : 0;
            }

            events.fire('edit.add', new SelectOp(splat, op, mask));
        });
    });

    // Size threshold selection - selects all points with total size (x+y+z) below the given threshold
    // This is useful for data cleanup by selecting tiny splats that contribute little to the visual quality
    events.on('select.sizeThreshold', (op: 'add'|'remove'|'set', threshold: number, direction: 'leq'|'geq' = 'leq') => {
        const splats = selectedSplats();
        if (!splats.length) {
            return;
        }

        const sizeThreshold = Math.max(0, Number.isFinite(threshold) ? threshold : 0);

        splats.forEach((splat) => {
            const sizeX = splat.splatData.getProp('scale_0') as Float32Array;
            const sizeY = splat.splatData.getProp('scale_1') as Float32Array;
            const sizeZ = splat.splatData.getProp('scale_2') as Float32Array;

            if (!sizeX || !sizeY || !sizeZ) {
                return;
            }

            const numSplats = splat.splatData.numSplats;
            const mask = new Uint8Array(numSplats);
            for (let i = 0; i < numSplats; i++) {
                // Convert scale values from log space to actual size and sum them
                const totalSize = Math.exp(sizeX[i]) + Math.exp(sizeY[i]) + Math.exp(sizeZ[i]);
                if (direction === 'leq') {
                    mask[i] = totalSize <= sizeThreshold ? 255 : 0;
                } else {
                    mask[i] = totalSize >= sizeThreshold ? 255 : 0;
                }
            }

            events.fire('edit.add', new SelectOp(splat, op, mask));
        });
    });

    // Re-center a fresh set of extracted gaussian positions around their own
    // bound center and return that center (world space). Positions are
    // subtracted in place so the new splat's entity can sit at the gaussians;
    // the gizmo (pivot origin 'center') then appears at the gaussians instead
    // of the world origin.
    const recenterExtractedPositions = (props: { name: string, storage: any }[], count: number): Vec3 | null => {
        const x = props.find(p => p.name === 'x')?.storage as Float32Array;
        const y = props.find(p => p.name === 'y')?.storage as Float32Array;
        const z = props.find(p => p.name === 'z')?.storage as Float32Array;
        if (!x || !y || !z || count === 0) return null;

        let minX = Infinity;
        let minY = Infinity;
        let minZ = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let maxZ = -Infinity;

        for (let i = 0; i < count; i++) {
            if (x[i] < minX) minX = x[i];
            if (x[i] > maxX) maxX = x[i];
            if (y[i] < minY) minY = y[i];
            if (y[i] > maxY) maxY = y[i];
            if (z[i] < minZ) minZ = z[i];
            if (z[i] > maxZ) maxZ = z[i];
        }

        if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(minZ)) return null;

        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        const cz = (minZ + maxZ) / 2;

        for (let i = 0; i < count; i++) {
            x[i] -= cx;
            y[i] -= cy;
            z[i] -= cz;
        }

        return new Vec3(cx, cy, cz);
    };

    const performSelectionFunc = (func: 'duplicate' | 'separate') => {
        const splats = selectedSplats();
        if (splats.length === 0) return;

        const splat = splats[0];
        const hasGaussianSelection = splat.numSelected > 0;

        // For separate, we need gaussian-level selection
        if (func === 'separate' && !hasGaussianSelection) return;

        const numSplats = splat.splatData.numSplats;
        const stateData = splat.splatData.getProp('state') as Uint8Array;

        // Count selected gaussians first (avoid building a huge index array)
        let selectedCount = 0;
        if (hasGaussianSelection && stateData) {
            for (let i = 0; i < numSplats; i++) {
                if (stateData[i] === State.selected) selectedCount++;
            }
        } else {
            selectedCount = numSplats;
        }

        if (selectedCount === 0) return;

        // Filter out internal properties (state, transform) — they reference
        // the source splat's palette and are invalid in the new splat.
        // This also reduces memory usage compared to copying all properties.
        const refProps = splat.splatData.getElement('vertex').properties;
        const internalProps = ['state', 'transform'];
        const props = refProps.filter(p => !internalProps.includes(p.name));

        const propNames = props.map(p => p.name);

        // Use SingleSplat to bake palette transforms into the extracted data.
        // Without this, palette transforms applied via point-cloud group
        // (move/rotate/scale) are lost — separating or duplicating would
        // revert gaussians to their pre-transform state. The entity transform
        // is kept separate so the copy can clone it and stay identical to the
        // source model (including gizmo position/orientation/scale).
        const singleSplat = new SingleSplat(propNames, { keepWorldTransform: true, skipPlyRotation: true });

        const extractedProps: typeof props = [];

        for (const prop of props) {
            const firstStorage = splat.splatData.getProp(prop.name);
            const Ctor = firstStorage.constructor as any;
            extractedProps.push({
                type: prop.type,
                name: prop.name,
                storage: new Ctor(selectedCount),
                byteSize: prop.byteSize
            });
        }

        let writeOffset = 0;
        for (let i = 0; i < numSplats; i++) {
            if (hasGaussianSelection && stateData && stateData[i] !== State.selected) continue;

            singleSplat.read(splat, i);

            for (let pi = 0; pi < extractedProps.length; pi++) {
                (extractedProps[pi].storage as any)[writeOffset] = singleSplat.data[propNames[pi]] ?? 0;
            }

            writeOffset++;
        }

        const extractedGSplatData = new GSplatData([{
            name: 'vertex',
            count: selectedCount,
            properties: extractedProps
        }]);

        // Gaussian data stays in the source's local space (palette transforms
        // already baked in), so clone the source entity transform to make the
        // copy a true duplicate — including its gizmo position/orientation/scale.
        const filename = `${removeExtension(splat.filename)}_${func}.ply`;
        const asset = new Asset(filename, 'gsplat', { url: `local-asset-${Date.now()}`, filename: filename });
        scene.app.assets.add(asset);
        asset.resource = new GSplatResource(scene.app.graphicsDevice, extractedGSplatData);
        const copy = new Splat(asset, new Quat());
        copy.entity.setLocalPosition(splat.entity.getLocalPosition().clone());
        copy.entity.setLocalRotation(splat.entity.getLocalRotation().clone());
        copy.entity.setLocalScale(splat.entity.getLocalScale().clone());

        if (func === 'separate') {
            editHistory.add(new MultiOp([
                new DeleteSelectionOp(splat),
                new AddSplatOp(scene, copy)
            ]));
        } else {
            editHistory.add(new AddSplatOp(scene, copy));
        }
    };

    // duplicate the current selection
    events.on('select.duplicate', () => {
        performSelectionFunc('duplicate');
    });

    events.on('select.separate', () => {
        performSelectionFunc('separate');
    });

    // Merge multiple selected splat files into one
    events.on('select.merge', async () => {
        const multiSelected = events.invoke('multiSplatSelection') as Splat[];
        if (multiSelected.length < 2) return;

        // Try backend merge first — C++ native engine avoids browser OOM for large files.
        // Requires: backend available + all splats have original PLY file paths.
        const backendAvailable = await BackendClient.isAvailable();
        const filePaths = multiSelected.map(s => s.originalFilePath).filter(Boolean) as string[];
        const allPly = filePaths.every(p => p.toLowerCase().endsWith('.ply') && !p.toLowerCase().endsWith('.compressed.ply'));

        if (backendAvailable && filePaths.length === multiSelected.length && allPly) {
            try {
                events.fire('startSpinner');
                const result = await BackendClient.mergePath(filePaths);
                console.log(`[merge] Backend merged ${result.count.toLocaleString()} Gaussians, ${(result.sizeBytes / (1024 * 1024)).toFixed(1)} MB`);

                // Download merged compressed-ply and import via standard pipeline
                const blob = await fetch(result.url).then(r => r.blob());
                const file = new File([blob], result.filename);
                const imported = await events.invoke('import', [{ filename: result.filename, contents: file }]) as Splat[];

                if (!imported || imported.length === 0) {
                    throw new Error('Import of merged file returned no splats');
                }

                const mergedSplat = imported[0];
                // Mark as merged (no permanent file path)
                mergedSplat.originalFilePath = null;

                const mergeOp = new MergeOp(scene, multiSelected, mergedSplat);
                await editHistory.add(mergeOp);

                events.fire('selection.clearMultiSplat');
                events.fire('selection', mergeOp.mergedSplat);
                events.fire('stopSpinner');
                return;
            } catch (err) {
                events.fire('stopSpinner');
                console.error('[merge] Backend merge failed, falling back to in-memory:', err);
                // Fall through to in-memory merge below
            }
        }

        // In-memory merge fallback — concatenate GSplatData properties directly.
        // Count only non-deleted gaussians from each splat.
        const totalCount = multiSelected.reduce((sum, s) => sum + s.numSplats, 0);
        const srcDatas = multiSelected.map(s => s.splatData);

        try {
            // Collect the property names from the first splat (they must share the same set)
            const refProps = srcDatas[0].getElement('vertex').properties;

            // Filter out internal properties (state, transform) and limit SH bands
            const internalProps = ['state', 'transform'];
            const maxSHBands = 3;
            const props = refProps
            .filter(p => !internalProps.includes(p.name))
            .filter((p) => {
                if (!p.name.startsWith('f_rest_')) {
                    return true;
                }
                const i = parseInt(p.name.slice(7), 10);
                return i < [0, 9, 24, 45][maxSHBands];
            });

            const propNames = props.map(p => p.name);

            // Build merged storage
            const singleSplat = new SingleSplat(propNames, { maxSHBands, keepWorldTransform: false, skipPlyRotation: true });
            const mergedProps: typeof props = [];

            for (let pi = 0; pi < props.length; pi++) {
                const refProp = props[pi];
                const name = refProp.name;
                const firstStorage = srcDatas[0].getProp(name);
                const Ctor = firstStorage.constructor as any;
                const mergedStorage = new Ctor(totalCount);

                mergedProps.push({
                    type: refProp.type,
                    name,
                    storage: mergedStorage,
                    byteSize: refProp.byteSize
                });
            }

            // Apply world transforms using SingleSplat and write to merged storage.
            // Skip deleted gaussians — they should not appear in the merged result.
            let writeOffset = 0;

            for (const splat of multiSelected) {
                const numSplats = splat.splatData.numSplats;
                const state = splat.splatData.getProp('state') as Uint8Array;

                for (let i = 0; i < numSplats; i++) {
                    if ((state[i] & State.deleted) !== 0) continue;

                    singleSplat.read(splat, i);

                    for (let pi = 0; pi < mergedProps.length; pi++) {
                        const prop = mergedProps[pi];
                        (prop.storage as any)[writeOffset] = singleSplat.data[prop.name] ?? 0;
                    }

                    writeOffset++;
                }
            }

            // Re-center the merged gaussians around their own bound center and
            // park the merged splat's entity there, so its gizmo appears at the
            // gaussians instead of the world origin.
            const mergedPosition = recenterExtractedPositions(mergedProps, totalCount);

            const mergedGSplatData = new GSplatData([{
                name: 'vertex',
                count: totalCount,
                properties: mergedProps
            }]);

            // Create Asset and Splat
            const firstName = removeExtension(multiSelected[0].filename);
            const mergedFilename = `${firstName}_merged.ply`;
            const asset = new Asset(mergedFilename, 'gsplat', { url: `local-asset-${Date.now()}`, filename: mergedFilename });
            scene.app.assets.add(asset);
            asset.resource = new GSplatResource(scene.app.graphicsDevice, mergedGSplatData);
            // Use identity rotation since all transforms are already baked into the merged data
            const mergedSplat = new Splat(asset, new Quat());
            if (mergedPosition) {
                mergedSplat.entity.setLocalPosition(mergedPosition);
            }

            const mergeOp = new MergeOp(scene, multiSelected, mergedSplat);
            await editHistory.add(mergeOp);

            // Clear multi-selection and select the new merged splat
            events.fire('selection.clearMultiSplat');
            events.fire('selection', mergeOp.mergedSplat);
        } catch (err) {
            if (err instanceof RangeError && err.message.includes('Array buffer allocation failed')) {
                await events.invoke('showPopup', {
                    type: 'error',
                    header: '合并失败',
                    message: `内存不足，无法合并 ${totalCount.toLocaleString()} 个高斯点。\n\n` +
                        '建议：\n' +
                        '1. 先分别导出两个模型为 PLY 文件，再用命令行工具合并\n' +
                        '2. 使用命令行工具精简模型后再合并'
                });
            } else {
                throw err;
            }
        }
    });

    events.on('select.subdivide', async () => {
        const [target] = selectedSplats();
        if (!target || target.scene !== scene) return;

        const subdivisionUnavailable = !!(
            target.lodEditLog || target.lccFilePath || target.pagedLodEditSession || target.pagedLodDescriptor
        );
        if (subdivisionUnavailable) {
            await events.invoke('showPopup', {
                type: 'info',
                header: localize('popup.subdivide.unavailable-header'),
                message: localize('popup.subdivide.unavailable-message')
            });
            return;
        }

        const requests = planSplatSubdivision(target.splatData);
        if (requests.length === 0) {
            await events.invoke('showPopup', {
                type: 'info',
                header: localize('popup.subdivide.empty-header'),
                message: localize('popup.subdivide.empty-message')
            });
            return;
        }

        const childCount = requests.length * 4;
        if (childCount > DECAL_SUBDIVISION_POINT_BUDGET) {
            const confirmation = await events.invoke('showPopup', {
                type: 'okcancel',
                header: localize('popup.subdivide.large-header'),
                message: localize('popup.subdivide.large-message')
            });
            if (confirmation?.action !== 'ok') return;
        }

        const originalSnapshot = {
            data: target.splatData,
            transformIndices: target.getTransformIndices(),
            soloMask: new Uint8Array(target.soloMaskData),
            desaturateMask: new Uint8Array(target.desaturateMaskData)
        };
        let replacementAttempted = false;
        let groupChanges: ReturnType<typeof buildSubdivisionGroupChanges> = [];

        events.fire('startSpinner');
        events.fire('spinnerText', localize('popup.subdivide.processing'));
        try {
            const subdivision = subdivideSplatData(
                originalSnapshot.data,
                requests,
                originalSnapshot.transformIndices,
                originalSnapshot.soloMask,
                originalSnapshot.desaturateMask
            );
            if (!subdivision) throw new Error('Subdivision produced no child Gaussians.');

            groupChanges = buildSubdivisionGroupChanges(
                events,
                target,
                originalSnapshot.data.numSplats,
                subdivision.data.numSplats,
                subdivision.childRanges
            );
            replacementAttempted = true;
            await target.replaceSplatData(subdivision);
            applySubdivisionGroups(events, target, groupChanges, true);
            if (target.scene !== scene) throw new Error('Subdivision target changed.');

            events.fire('edit.add', new SplatSubdivideOp({
                splat: target,
                events,
                structuralIndices: subdivision.structuralIndices,
                beforeStates: subdivision.beforeStates,
                afterStates: subdivision.afterStates,
                groupChanges
            }), true);
        } catch (error) {
            if (replacementAttempted && target.scene === scene) {
                try {
                    await target.replaceSplatData(originalSnapshot);
                    applySubdivisionGroups(events, target, groupChanges, false);
                } catch (rollbackError) {
                    console.error('[Subdivide] Failed to roll back subdivision', rollbackError);
                }
            }
            console.error('[Subdivide] Failed to subdivide splat', error);
            await events.invoke('showPopup', {
                type: 'error',
                header: localize('popup.subdivide.error-header'),
                message: error instanceof Error ? error.message : String(error)
            });
        } finally {
            events.fire('stopSpinner');
        }
    });

    events.on('scene.reset', () => {
        selectedSplats().forEach((splat) => {
            editHistory.add(new ResetOp(splat));
        });
    });

    // camera mode (visual: centers/rings)

    let activeMode = 'splat';

    const setCameraMode = (mode: string) => {
        if (mode !== activeMode) {
            activeMode = mode;
            events.fire('camera.mode', activeMode);
        }
    };

    events.function('camera.mode', () => {
        return activeMode;
    });

    events.on('camera.setMode', (mode: string) => {
        setCameraMode(mode);
    });

    events.on('camera.toggleMode', () => {
        setCameraMode(events.invoke('camera.mode') === 'centers' ? 'rings' : 'centers');
    });

    events.on('camera.cycleMode', () => {
        const modes = ['splat', 'centers', 'rings'];
        const currentIndex = modes.indexOf(activeMode);
        const nextIndex = (currentIndex + 1) % modes.length;
        setCameraMode(modes[nextIndex]);
    });

    // camera control mode (orbit/fly)

    let controlMode: 'orbit' | 'fly' = 'orbit';

    const setControlMode = (mode: 'orbit' | 'fly') => {
        if (mode !== controlMode) {
            controlMode = mode;
            scene.camera.controlMode = mode;
            events.fire('camera.controlMode', controlMode);
        }
    };

    events.function('camera.controlMode', () => {
        return controlMode;
    });

    events.on('camera.setControlMode', (mode: 'orbit' | 'fly') => {
        setControlMode(mode);
    });

    events.on('camera.toggleControlMode', () => {
        setControlMode(controlMode === 'orbit' ? 'fly' : 'orbit');
    });

    // camera overlay

    let cameraOverlay = scene.config.camera.overlay;

    const setCameraOverlay = (enabled: boolean) => {
        if (enabled !== cameraOverlay) {
            cameraOverlay = enabled;
            events.fire('camera.overlay', cameraOverlay);
        }
    };

    events.function('camera.overlay', () => {
        return cameraOverlay;
    });

    events.on('camera.setOverlay', (value: boolean) => {
        setCameraOverlay(value);
    });

    events.on('camera.toggleOverlay', () => {
        setCameraOverlay(!events.invoke('camera.overlay'));
    });

    // splat size

    let splatSize = 2;

    const setSplatSize = (value: number) => {
        if (value !== splatSize) {
            splatSize = value;
            events.fire('camera.splatSize', splatSize);
        }
    };

    events.function('camera.splatSize', () => {
        return splatSize;
    });

    events.on('camera.setSplatSize', (value: number) => {
        setSplatSize(value);
    });

    // camera fly speed

    const setFlySpeed = (value: number) => {
        if (value !== scene.camera.flySpeed) {
            scene.camera.flySpeed = value;
            events.fire('camera.flySpeed', value);
        }
    };

    events.function('camera.flySpeed', () => {
        return scene.camera.flySpeed;
    });

    events.on('camera.setFlySpeed', (value: number) => {
        setFlySpeed(value);
    });

    // outline selection

    let outlineSelection = false;

    const setOutlineSelection = (value: boolean) => {
        if (value !== outlineSelection) {
            outlineSelection = value;
            events.fire('view.outlineSelection', outlineSelection);
        }
    };

    events.function('view.outlineSelection', () => {
        return outlineSelection;
    });

    events.on('view.setOutlineSelection', (value: boolean) => {
        setOutlineSelection(value);
    });

    // view spherical harmonic bands

    let viewBands = scene.config.show.shBands;

    const setViewBands = (value: number) => {
        if (value !== viewBands) {
            viewBands = value;
            events.fire('view.bands', viewBands);
        }
    };

    events.function('view.bands', () => {
        return viewBands;
    });

    events.on('view.setBands', (value: number) => {
        setViewBands(value);
    });

    // view depth cycle length (fmod range for depth mode)

    let depthCycleLength = 50;

    const setDepthCycleLength = (value: number) => {
        const clamped = Math.max(1, Math.min(100, Math.round(value)));
        if (clamped !== depthCycleLength) {
            depthCycleLength = clamped;
            events.fire('view.depthCycleLength', depthCycleLength);
            scene.forceRender = true;
        }
    };

    events.function('view.depthCycleLength', () => {
        return depthCycleLength;
    });

    events.on('view.setDepthCycleLength', (value: number) => {
        setDepthCycleLength(value);
    });

    events.fire('view.depthCycleLength', depthCycleLength);

    // centers gaussian color toggle
    let centersUseGaussianColor = false;
    events.function('view.centersUseGaussianColor', () => centersUseGaussianColor);
    events.on('view.setCentersUseGaussianColor', (value: boolean) => {
        centersUseGaussianColor = value;
        events.fire('view.centersUseGaussianColor', value);
    });

    events.function('camera.getPose', () => {
        const camera = scene.camera;
        const position = camera.position;
        const focalPoint = camera.focalPoint;
        return {
            position: { x: position.x, y: position.y, z: position.z },
            target: { x: focalPoint.x, y: focalPoint.y, z: focalPoint.z },
            fov: camera.fov
        };
    });

    events.on('camera.setPose', (pose: { position: Vec3, target: Vec3, fov?: number }, speed = 1) => {
        // assign fov before setPose so distance is computed using the new fovFactor
        if (pose.fov !== undefined) {
            scene.camera.fov = pose.fov;
            events.fire('camera.fov', pose.fov);
        }
        scene.camera.setPose(pose.position, pose.target, speed);
    });

    // hack: fire events to initialize UI
    events.fire('camera.fov', scene.camera.fov);
    events.fire('camera.overlay', cameraOverlay);
    events.fire('view.bands', viewBands);

    // doc serialization
    events.function('docSerialize.view', () => {
        const packC = (c: Color) => [c.r, c.g, c.b, c.a];
        return {
            bgColor: packC(events.invoke('bgClr')),
            selectedColor: packC(events.invoke('selectedClr')),
            unselectedColor: packC(events.invoke('unselectedClr')),
            lockedColor: packC(events.invoke('lockedClr')),
            shBands: events.invoke('view.bands'),
            centersSize: events.invoke('camera.splatSize'),
            outlineSelection: events.invoke('view.outlineSelection'),
            showGrid: events.invoke('grid.visible'),
            gridPlane: events.invoke('grid.plane'),
            showBound: events.invoke('camera.bound'),
            showBoundDimensions: events.invoke('camera.boundDimensions'),
            showCameraPoses: events.invoke('camera.showPoses'),
            flySpeed: events.invoke('camera.flySpeed'),
            fovDolly: events.invoke('camera.fovDolly'),
            depthCycleLength: events.invoke('view.depthCycleLength')
        };
    });

    events.function('docDeserialize.view', (docView: any) => {
        events.fire('setBgClr', new Color(docView.bgColor));
        events.fire('setSelectedClr', new Color(docView.selectedColor));
        events.fire('setUnselectedClr', new Color(docView.unselectedColor));
        events.fire('setLockedClr', new Color(docView.lockedColor));
        events.fire('view.setBands', docView.shBands);
        events.fire('camera.setSplatSize', docView.centersSize);
        events.fire('view.setOutlineSelection', docView.outlineSelection);
        events.fire('grid.setVisible', docView.showGrid);
        events.fire('grid.setPlane', docView.gridPlane ?? 'xz');
        events.fire('camera.setBound', docView.showBound);
        events.fire('camera.setBoundDimensions', docView.showBoundDimensions ?? false);
        events.fire('camera.setShowPoses', docView.showCameraPoses ?? false);
        events.fire('camera.setFlySpeed', docView.flySpeed);
        events.fire('camera.setFovDolly', docView.fovDolly ?? false);
        if (docView.depthCycleLength !== undefined) {
            events.fire('view.setDepthCycleLength', docView.depthCycleLength);
        }
    });

    // --- view preferences persistence (localStorage) ---

    // debounced save to avoid frequent writes during slider drags
    let prefsSaveTimer: ReturnType<typeof setTimeout> | null = null;
    const scheduleSavePrefs = () => {
        if (prefsSaveTimer) clearTimeout(prefsSaveTimer);
        prefsSaveTimer = setTimeout(() => {
            saveViewPrefs(collectViewPrefs(events));
        }, 300);
    };

    // listen to all view-option state-change events
    [
        'bgClr', 'selectedClr', 'unselectedClr', 'lockedClr',
        'camera.tonemapping', 'camera.fov', 'camera.fovDolly', 'view.bands',
        'camera.flySpeed', 'camera.splatSize',
        'view.centersUseGaussianColor', 'view.outlineSelection', 'grid.plane',
        'camera.boundDimensions',
        'camera.showPoses'
    ].forEach((eventName) => {
        events.on(eventName, scheduleSavePrefs);
    });

    // load and apply saved preferences on startup
    const prefs = loadViewPrefs();
    if (prefs) {
        applyViewPrefs(events, prefs);
    }
};

export { registerEditorEvents };
