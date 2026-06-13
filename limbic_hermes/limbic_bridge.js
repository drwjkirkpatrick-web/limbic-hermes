// limbic_hermes/limbic_bridge.js
// ==============================
// Browser-side helper for the Hermes aquarium dashboard (or any web UI).
//
// Polls a small state JSON endpoint (or reads localStorage) and exposes the
// limbic VAD vector as easy knobs for visualizations.
//
// Usage:
//   const lb = new LimbicBridge({
//       source: 'http://localhost:8888/limbic_state.json',
//       localStorageKey: 'hermes_limbic_state',
//       intervalMs: 1000
//   });
//   lb.onUpdate = (state) => {
//       fish.setAffect(state.dominant_affect, state.vad);
//   };
//   lb.start();

class LimbicBridge {
  constructor(options = {}) {
    this.source = options.source || null; // e.g. '/limbic_state.json'
    this.localStorageKey = options.localStorageKey || 'hermes_limbic_state';
    this.intervalMs = options.intervalMs || 1000;
    this.state = null;
    this.interval = null;
    this.onUpdate = options.onUpdate || (() => {});
  }

  readLocal() {
    try {
      const raw = localStorage.getItem(this.localStorageKey);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  writeLocal(state) {
    try {
      localStorage.setItem(this.localStorageKey, JSON.stringify(state));
    } catch (e) {
      // ignore (e.g. storage disabled)
    }
  }

  async fetchRemote() {
    if (!this.source) return null;
    try {
      const res = await fetch(this.source, { cache: 'no-store' });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  async tick() {
    let state = await this.fetchRemote();
    if (!state) {
      state = this.readLocal();
    }
    if (state) {
      this.state = state;
      this.writeLocal(state);
      this.onUpdate(state);
    }
  }

  start() {
    if (this.interval) return this;
    this.tick();
    this.interval = setInterval(() => this.tick(), this.intervalMs);
    return this;
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    return this;
  }
}

// Convenience mapping for the aquarium angelfish
function affectToFishState(state) {
  const affect = state.dominant_affect;
  const v = state.vad?.valence || 0;
  const a = state.vad?.arousal || 0;
  const d = state.vad?.dominance || 0.5;

  const fishState = {
    colorTemp: 0.5 + v * 0.5,          // 0 cool/blue, 1 warm/yellow
    finSpeed: 0.5 + a * 1.5,           // beats per animation
    posture: d,                         // 0 drooping, 1 upright
    glow: 1.0 - (state.drive?.rest_need || 0) * 0.6,
  };

  // Map dominant affect label to existing aquarium states
  const stateMap = {
    calm: 'idle',
    neutral: 'idle',
    confident: 'active',
    hopeful: 'active',
    activated: 'thinking',
    anxious: 'error',
    irritable: 'alert',
    sad: 'sleeping',
  };
  fishState.agentState = stateMap[affect] || 'idle';
  return fishState;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LimbicBridge, affectToFishState };
}
