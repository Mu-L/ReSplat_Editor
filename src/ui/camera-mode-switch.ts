import { Button, Container, Label, SliderInput } from '@playcanvas/pcui';

import { Events } from '../events';
import { ShortcutManager } from '../shortcut-manager';
import { localize, formatTooltipWithShortcut } from './localization';
import flyCameraSvg from './svg/fly-camera.svg';
import translateSvg from './svg/move.svg';
import orbitCameraSvg from './svg/orbit-camera.svg';
import rotateSvg from './svg/rotate.svg';
import scaleSvg from './svg/scale.svg';
import { Tooltips } from './tooltips';

const createSvg = (svgString: string) => {
    const decodedStr = decodeURIComponent(svgString.substring('data:image/svg+xml,'.length));
    return new DOMParser().parseFromString(decodedStr, 'image/svg+xml').documentElement;
};

class CameraModeSwitch extends Container {
    constructor(events: Events, tooltips: Tooltips, args = {}) {
        args = {
            ...args,
            id: 'camera-mode-switch'
        };

        super(args);

        this.dom.addEventListener('pointerdown', (event) => {
            event.stopPropagation();
        });

        const orbitMode = new Button({
            id: 'camera-mode-orbit',
            class: ['camera-mode-btn', 'active']
        });

        const flyMode = new Button({
            id: 'camera-mode-fly',
            class: 'camera-mode-btn'
        });

        orbitMode.dom.appendChild(createSvg(orbitCameraSvg));
        flyMode.dom.appendChild(createSvg(flyCameraSvg));

        this.append(orbitMode);
        this.append(flyMode);

        const transformTools = new Container({
            id: 'transform-tool-switch'
        });
        const translate = new Button({
            id: 'transform-tool-translate',
            class: ['camera-mode-btn', 'transform-tool-btn']
        });
        const rotate = new Button({
            id: 'transform-tool-rotate',
            class: ['camera-mode-btn', 'transform-tool-btn']
        });
        const scale = new Button({
            id: 'transform-tool-scale',
            class: ['camera-mode-btn', 'transform-tool-btn']
        });

        translate.dom.appendChild(createSvg(translateSvg));
        rotate.dom.appendChild(createSvg(rotateSvg));
        scale.dom.appendChild(createSvg(scaleSvg));
        transformTools.append(translate);
        transformTools.append(rotate);
        transformTools.append(scale);
        this.append(transformTools);

        const shortcutManager: ShortcutManager = events.invoke('shortcutManager');
        const tooltip = (localeKey: string, shortcutId?: string) => {
            const text = localize(localeKey);
            if (shortcutId) {
                const shortcut = shortcutManager.formatShortcut(shortcutId);
                if (shortcut) {
                    return formatTooltipWithShortcut(text, shortcut);
                }
            }
            return text;
        };

        tooltips.register(orbitMode, tooltip('tooltip.right-toolbar.orbit-camera', 'camera.toggleControlMode'), 'bottom');
        tooltips.register(flyMode, tooltip('tooltip.right-toolbar.fly-camera', 'camera.toggleControlMode'), 'bottom');
        tooltips.register(translate, tooltip('tooltip.left-toolbar.translate', 'tool.move'), 'bottom');
        tooltips.register(rotate, tooltip('tooltip.left-toolbar.rotate', 'tool.rotate'), 'bottom');
        tooltips.register(scale, tooltip('tooltip.left-toolbar.scale', 'tool.scale'), 'bottom');

        translate.on('click', () => events.fire('tool.move'));
        rotate.on('click', () => events.fire('tool.rotate'));
        scale.on('click', () => events.fire('tool.scale'));

        events.on('tool.activated', (toolName: string) => {
            translate.class[toolName === 'move' ? 'add' : 'remove']('active');
            rotate.class[toolName === 'rotate' ? 'add' : 'remove']('active');
            scale.class[toolName === 'scale' ? 'add' : 'remove']('active');
        });

        events.on('mode.changed', (mode: 'edit' | 'paint') => {
            const painting = mode === 'paint';
            transformTools.hidden = painting;
            this.dom.classList.toggle('transform-tools-hidden', painting);
        });

        orbitMode.on('click', () => {
            events.fire('camera.setControlMode', 'orbit');
            showSpeedPopup();
        });
        flyMode.on('click', () => {
            events.fire('camera.setControlMode', 'fly');
            showSpeedPopup();
        });

        // fly speed popup bar
        const speedBar = new Container({
            id: 'fly-speed-popup'
        });

        const speedLabel = new Label({
            text: localize('panel.view-options.fly-speed'),
            class: 'fly-speed-popup-label'
        });

        const speedSlider = new SliderInput({
            class: 'fly-speed-popup-slider',
            min: 0.1,
            max: 100,
            sliderMax: 100,
            precision: 1,
            value: 1
        });

        speedBar.append(speedLabel);
        speedBar.append(speedSlider);
        speedBar.dom.style.display = 'none';
        this.append(speedBar);

        let hideTimeout: ReturnType<typeof setTimeout> | null = null;
        let currentControlMode: 'orbit' | 'fly' = 'orbit';
        let suppressChange = false;
        let speedPopupHovered = false;

        const resetSpeedHideTimeout = () => {
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }
            hideTimeout = setTimeout(() => {
                hideTimeout = null;
                if (!speedPopupHovered) {
                    const active = document.activeElement;
                    if (!speedBar.dom.contains(active)) {
                        speedBar.dom.style.display = 'none';
                    } else {
                        resetSpeedHideTimeout();
                    }
                }
            }, 1000);
        };

        const showSpeedPopup = () => {
            speedBar.dom.style.display = '';
            resetSpeedHideTimeout();
        };

        const hideSpeedPopup = () => {
            speedBar.dom.style.display = 'none';
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }
        };

        // Pause hide timer while hovering over popup
        speedBar.dom.addEventListener('mouseenter', () => {
            speedPopupHovered = true;
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }
        });
        speedBar.dom.addEventListener('mouseleave', () => {
            speedPopupHovered = false;
            resetSpeedHideTimeout();
        });

        // Reset timer on any interaction with the popup
        speedBar.dom.addEventListener('pointerdown', () => {
            resetSpeedHideTimeout();
        });
        speedBar.dom.addEventListener('pointermove', (e: PointerEvent) => {
            if (e.buttons) {
                resetSpeedHideTimeout();
            }
        });
        speedBar.dom.addEventListener('input', () => {
            resetSpeedHideTimeout();
        });
        speedBar.dom.addEventListener('focusin', () => {
            resetSpeedHideTimeout();
        });

        events.on('camera.controlMode', (mode: 'orbit' | 'fly') => {
            currentControlMode = mode;
            if (mode === 'orbit') {
                hideSpeedPopup();
            }
            orbitMode.class[mode === 'orbit' ? 'add' : 'remove']('active');
            flyMode.class[mode === 'fly' ? 'add' : 'remove']('active');
        });

        events.on('camera.flySpeed', (value: number) => {
            suppressChange = true;
            speedSlider.value = value;
            speedSlider.step = value > 20 ? 10 : 1;
            setTimeout(() => {
                suppressChange = false;
            }, 0);
            if (currentControlMode !== 'fly') return;
            showSpeedPopup();
        });

        // 拖拽跟踪：拖拽中自由移动，松手后才snap
        let isDragging = false;
        const sliderDom = (speedSlider as any)._domSlider;
        if (sliderDom) {
            sliderDom.addEventListener('pointerdown', () => {
                isDragging = true;
            });
            sliderDom.addEventListener('pointerup', () => {
                isDragging = false;
            });
        }

        speedSlider.on('change', (value: number) => {
            if (suppressChange) return;
            // 拖拽中自由移动，不snap
            if (!isDragging && value > 20) {
                const snapped = Math.round(value / 10) * 10;
                if (snapped !== value) {
                    suppressChange = true;
                    speedSlider.value = snapped;
                    suppressChange = false;
                    events.fire('camera.setFlySpeed', snapped);
                    resetSpeedHideTimeout();
                    return;
                }
            }
            events.fire('camera.setFlySpeed', value);
            resetSpeedHideTimeout();
        });
    }
}

export { CameraModeSwitch };
