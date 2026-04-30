"""Shared browser init scripts for safer, more stable benchmark execution."""

from __future__ import annotations


PAGE_SAFETY_INIT_SCRIPT = r"""
(() => {
  const noop = () => {};
  const makeParam = (value) => ({
    value,
    setValueAtTime: noop,
    linearRampToValueAtTime: noop,
    exponentialRampToValueAtTime: noop,
    cancelScheduledValues: noop,
  });
  const makeNode = () => ({
    connect() { return this; },
    disconnect: noop,
    start: noop,
    stop: noop,
    addEventListener: noop,
    removeEventListener: noop,
    gain: makeParam(1),
    frequency: makeParam(440),
    detune: makeParam(0),
    Q: makeParam(1),
    playbackRate: makeParam(1),
    buffer: null,
    loop: false,
  });

  class SilentAudioContext {
    constructor() {
      this.state = 'running';
      this.currentTime = 0;
      this.sampleRate = 44100;
      this.destination = makeNode();
    }
    createOscillator() { return makeNode(); }
    createGain() { return makeNode(); }
    createBufferSource() { return makeNode(); }
    createBiquadFilter() { return makeNode(); }
    createAnalyser() { return makeNode(); }
    createStereoPanner() { return makeNode(); }
    createDynamicsCompressor() { return makeNode(); }
    createDelay() { return makeNode(); }
    createConvolver() { return makeNode(); }
    createPeriodicWave() { return {}; }
    createWaveShaper() { return makeNode(); }
    createScriptProcessor() { return makeNode(); }
    createBuffer(channels, length, sampleRate) {
      return {
        numberOfChannels: channels,
        length,
        sampleRate,
        getChannelData() { return new Float32Array(length); },
      };
    }
    resume() { this.state = 'running'; return Promise.resolve(); }
    suspend() { this.state = 'suspended'; return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
  }

  try {
    Object.defineProperty(window, 'AudioContext', {
      configurable: true,
      writable: true,
      value: SilentAudioContext,
    });
    Object.defineProperty(window, 'webkitAudioContext', {
      configurable: true,
      writable: true,
      value: SilentAudioContext,
    });
  } catch (_) {}

  try {
    if (window.HTMLMediaElement && window.HTMLMediaElement.prototype) {
      window.HTMLMediaElement.prototype.play = function play() {
        return Promise.resolve();
      };
    }
  } catch (_) {}
})();
"""


async def install_page_safety(target) -> None:
    """Install a lightweight init script on a Playwright page or context."""
    await target.add_init_script(PAGE_SAFETY_INIT_SCRIPT)
