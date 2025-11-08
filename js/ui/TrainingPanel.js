import { BaseModule } from '../core/BaseModule.js';
import { FEATURE_KEYS } from '../audio/FeatureExtractor.js';
import { PARTICLE_PARAMETER_TARGETS } from '../ml/MLPTrainingTargets.js';
import { DEFAULT_CORRELATION, sanitizeCorrelationList } from '../ml/MLPTrainingUtils.js';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
const statusLabels = {
  idle: 'Idle',
  running: 'Training',
  paused: 'Paused',
  completed: 'Ready',
  aborted: 'Aborted',
  error: 'Error',
};

const POLARITY_OPTIONS = [
  { id: 'direct', label: 'Direct' },
  { id: 'inverse', label: 'Inverse' },
];

const createElement = (tag, className = '', text = '') => {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text) el.textContent = text;
  return el;
};

export class TrainingPanel extends BaseModule {
  constructor(options = {}) {
    super('TrainingPanel');
    this.trainingManager = options.trainingManager || null;
    this.mlpModel = options.mlpModel || null;
    this.mlpController = options.mlpController || null;
    this.featureKeys = options.featureKeys || FEATURE_KEYS;
    this.positionalFeatures = Array.isArray(options.positionalFeatures) ? options.positionalFeatures : [];
    const positionalLabels = this.positionalFeatures.reduce((acc, feature) => {
      if (feature?.id) {
        acc[feature.id] = feature.label || feature.id;
      }
      return acc;
    }, {});
    this.featureLabels = {
      ...positionalLabels,
      ...(options.featureLabels || {}),
    };
    this.targetLabels = PARTICLE_PARAMETER_TARGETS.reduce((acc, target) => {
      acc[target.id] = target.label;
      return acc;
    }, {});
    this.maxCorrelations = Number.isFinite(options.maxCorrelations)
      ? options.maxCorrelations
      : Infinity;
    this.state = {
      correlations: [],
      status: 'idle',
      epoch: 0,
      epochs: options.trainingOptions?.epochs || 1,
      loss: null,
      metadata: null,
      trainingOptions: {
        epochs: 24,
        batchSize: 128,
        learningRate: 0.0015,
        sampleCount: 4096,
        noise: 0.04,
        ...(options.trainingOptions || {}),
      },
      exportName: '',
      exportNotes: '',
    };
    this.root = null;
    this.correlationListEl = null;
    this.statusEl = null;
    this.progressEl = null;
    this.lossEl = null;
    this.resultEl = null;
    this.errorEl = null;
    this.addButton = null;
    this.buttons = {};
    this.fileInput = null;
    this.unsubscribes = [];
    this._initStateFromManager();
    this.showTrainingControls = options.showTrainingControls !== false;
    this.showHeader = options.showHeader !== false;
    this.showStatusBar = options.showStatusBar !== false;
    this.showResultSection = options.showResultSection !== false;
    this.showExportSection = options.showExportSection !== false;
    this.embedded = Boolean(options.embedded);
    this.mountTarget = options.mountTarget || null;
  }

  init() {
    if (this.initialized || !this.trainingManager || typeof document === 'undefined') {
      if (!this.trainingManager) {
        console.warn('[TrainingPanel] Training manager is not available; panel disabled.');
      }
      return this;
    }
    this.root = createElement('aside', 'training-panel');
    if (this.embedded) {
      this.root.classList.add('training-panel--embedded');
    }
    this.root.setAttribute('aria-live', 'polite');
    this.root.innerHTML = '';

    const children = [];
    if (this.showHeader) {
      const header = createElement('header', 'training-panel__header');
      header.appendChild(createElement('h3', '', 'In-Browser Training'));
      header.appendChild(createElement('p', '', 'Map audio features to particle parameters, then train & export custom models.'));
      children.push(header);
    }

    if (this.showStatusBar) {
      const statusBar = createElement('div', 'training-panel__status');
      this.statusEl = createElement('span', 'training-panel__status-text', 'Idle');
      this.progressEl = createElement('div', 'training-panel__progress');
      const progressBar = createElement('div', 'training-panel__progress-bar');
      this.progressEl.appendChild(progressBar);
      this.lossEl = createElement('span', 'training-panel__loss', '');
      statusBar.append(this.statusEl, this.progressEl, this.lossEl);
      children.push(statusBar);
    } else {
      this.statusEl = null;
      this.progressEl = null;
      this.lossEl = null;
    }

    children.push(this._buildCorrelationSection());
    if (this.showTrainingControls) {
      children.push(this._buildTrainingSection());
    }
    if (this.showResultSection) {
      children.push(this._buildResultSection());
    }
    if (this.showExportSection) {
      children.push(this._buildExportSection());
    }

    this.errorEl = createElement('p', 'training-panel__error');
    this.errorEl.hidden = true;
    children.push(this.errorEl);

    this.root.append(...children);
    const mountTarget = this.mountTarget || document.body;
    mountTarget.appendChild(this.root);

    this._bindTrainingEvents();
    this._renderCorrelations();
    this._syncStatus();

    super.init();
    return this;
  }

  dispose() {
    this.unsubscribes.forEach((fn) => fn());
    this.unsubscribes.length = 0;
    if (this.root) {
      this.root.remove();
      this.root = null;
    }
    super.dispose();
  }

  _initStateFromManager() {
    if (!this.trainingManager) {
      this.state.correlations = [DEFAULT_CORRELATION(this.featureKeys[0], PARTICLE_PARAMETER_TARGETS[0].id)];
      return;
    }
    const existing = this.trainingManager.getCorrelations?.() || [];
    const sanitized = sanitizeCorrelationList(
      existing.length ? existing : [DEFAULT_CORRELATION(this.featureKeys[0], PARTICLE_PARAMETER_TARGETS[0].id)],
      this.featureKeys,
      this._getSanitizeOptions(),
    );
    this.state.correlations = sanitized;
    const opt = this.trainingManager.getTrainingOptions?.();
    if (opt) {
      this.state.trainingOptions = { ...this.state.trainingOptions, ...opt };
    }
    this.trainingManager.setCorrelations(this.state.correlations);
  }

  _buildCorrelationSection() {
    const section = createElement('section', 'training-panel__section');
    section.appendChild(createElement('h4', '', 'Feature ↔ Particle Correlations'));
    section.appendChild(
      createElement(
        'p',
        'training-panel__description',
        'Choose which audio features drive particle parameters and how strongly they correlate.',
      ),
    );

    this.correlationListEl = createElement('div', 'training-panel__correlation-list');
    section.appendChild(this.correlationListEl);

    const footer = createElement('div', 'training-panel__correlation-actions');
    this.addButton = createElement('button', 'training-panel__btn training-panel__btn--ghost', 'Add correlation');
    this.addButton.type = 'button';
    this.addButton.addEventListener('click', () => this._handleAddCorrelation());
    footer.appendChild(this.addButton);
    section.appendChild(footer);

    return section;
  }

  _buildTrainingSection() {
    const section = createElement('section', 'training-panel__section');
    section.appendChild(createElement('h4', '', 'Training Controls'));

    const grid = createElement('div', 'training-panel__grid');
    const options = [
      { key: 'epochs', label: 'Epochs', min: 1, max: 200, step: 1 },
      { key: 'batchSize', label: 'Batch size', min: 8, max: 1024, step: 8 },
      { key: 'sampleCount', label: 'Sample count', min: 512, max: 16384, step: 256 },
      { key: 'learningRate', label: 'Learning rate', min: 0.0001, max: 0.01, step: 0.0001, precision: 5 },
      { key: 'noise', label: 'Noise', min: 0, max: 0.25, step: 0.005, precision: 3 },
    ];

    options.forEach((option) => {
      const field = createElement('label', 'training-panel__field');
      field.textContent = option.label;
      const input = document.createElement('input');
      input.type = 'number';
      input.step = option.step;
      input.min = option.min;
      input.max = option.max;
      input.value = this.state.trainingOptions[option.key];
      input.addEventListener('input', () => {
        const raw = Number(input.value);
        const value = clamp(
          Number.isFinite(raw) ? raw : this.state.trainingOptions[option.key],
          option.min,
          option.max,
        );
        this.state.trainingOptions[option.key] =
          option.precision != null ? Number(value.toFixed(option.precision)) : value;
        this.trainingManager.updateTrainingOptions({ [option.key]: this.state.trainingOptions[option.key] });
      });
      field.appendChild(input);
      grid.appendChild(field);
    });

    const controls = createElement('div', 'training-panel__actions');
    this.buttons.random = this._createActionButton('Random model', 'ghost', () => this._handleRandomModel());
    this.buttons.start = this._createActionButton('Train', 'primary', () => this._handleStart());
    this.buttons.finetune = this._createActionButton('Finetune', 'ghost', () => this._handleFinetune());
    this.buttons.pause = this._createActionButton('Pause', 'ghost', () => this.trainingManager.pauseTraining());
    this.buttons.resume = this._createActionButton('Resume', 'ghost', () => this.trainingManager.resumeTraining());
    this.buttons.abort = this._createActionButton('Abort & Apply', 'danger', () => this.trainingManager.abortTraining());
    controls.append(
      this.buttons.random,
      this.buttons.start,
      this.buttons.finetune,
      this.buttons.pause,
      this.buttons.resume,
      this.buttons.abort,
    );

    section.append(grid, controls);
    return section;
  }

  _buildResultSection() {
    const section = createElement('section', 'training-panel__section');
    section.appendChild(createElement('h4', '', 'Training Result'));
    this.resultEl = createElement('div', 'training-panel__results');
    this.resultEl.textContent = 'No training run yet.';
    section.appendChild(this.resultEl);
    return section;
  }

  _buildExportSection() {
    const section = createElement('section', 'training-panel__section');
    section.appendChild(createElement('h4', '', 'Model Export / Import'));

    const metaGrid = createElement('div', 'training-panel__grid');
    const nameField = createElement('label', 'training-panel__field');
    nameField.textContent = 'Snapshot label';
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.placeholder = 'e.g. Midnight Bloom v1';
    nameInput.addEventListener('input', () => {
      this.state.exportName = nameInput.value.trim();
    });
    nameField.appendChild(nameInput);

    const notesField = createElement('label', 'training-panel__field');
    notesField.textContent = 'Notes';
    const notesInput = document.createElement('textarea');
    notesInput.rows = 2;
    notesInput.placeholder = 'Describe the vibe or track references...';
    notesInput.addEventListener('input', () => {
      this.state.exportNotes = notesInput.value.trim();
    });
    notesField.appendChild(notesInput);

    metaGrid.append(nameField, notesField);

    const actions = createElement('div', 'training-panel__actions');
    const exportBtn = this._createActionButton('Export model', 'primary', () => this._handleExport());
    const importBtn = this._createActionButton('Import model', 'ghost', () => this._handleImportClick());
    actions.append(exportBtn, importBtn);

    this.fileInput = document.createElement('input');
    this.fileInput.type = 'file';
    this.fileInput.accept = 'application/json';
    this.fileInput.addEventListener('change', (event) => this._handleImport(event));

    section.append(metaGrid, actions, this.fileInput);
    this.fileInput.hidden = true;
    return section;
  }

  _createActionButton(label, variant, handler) {
    const button = createElement('button', `training-panel__btn training-panel__btn--${variant}`, label);
    button.type = 'button';
    button.addEventListener('click', handler);
    return button;
  }

  _bindTrainingEvents() {
    const events = ['state', 'progress', 'result', 'error'];
    events.forEach((event) => {
      const unsubscribe = this.trainingManager.on(event, (payload) => this._handleTrainingEvent(event, payload));
      this.unsubscribes.push(unsubscribe);
    });
  }

  _handleTrainingEvent(event, payload) {
    switch (event) {
      case 'state':
      case 'progress':
        this.state = { ...this.state, ...payload };
        this._syncStatus(payload);
        break;
      case 'result':
        if (payload?.metadata) {
          this.state.metadata = payload.metadata;
          this._renderResult(payload.metadata);
        }
        this._syncStatus(payload);
        break;
      case 'error':
        this._showError(payload?.message || 'Training failed.');
        break;
      default:
        break;
    }
  }

  _handleAddCorrelation() {
    if (this.state.correlations.length >= this.maxCorrelations) {
      return;
    }
    const feature = this.featureKeys.length
      ? this.featureKeys[this.state.correlations.length % this.featureKeys.length]
      : this.positionalFeatures[0]?.id || this.featureKeys[0];
    const target = PARTICLE_PARAMETER_TARGETS[this.state.correlations.length % PARTICLE_PARAMETER_TARGETS.length];
    const next = DEFAULT_CORRELATION(feature, target.id);
    this.state.correlations = sanitizeCorrelationList(
      [...this.state.correlations, next],
      this.featureKeys,
      this._getSanitizeOptions(),
    );
    this._persistCorrelations();
    this._renderCorrelations();
  }

  _renderCorrelations() {
    if (!this.correlationListEl) return;
    this.correlationListEl.innerHTML = '';
    if (!this.state.correlations.length) {
      this.correlationListEl.appendChild(createElement('p', 'training-panel__empty', 'No correlations configured.'));
    } else {
      this.state.correlations.forEach((corr, index) => {
        this.correlationListEl.appendChild(this._createCorrelationRow(corr, index));
      });
    }
    if (this.addButton) {
      this.addButton.disabled = this.state.correlations.length >= this.maxCorrelations;
    }
  }

  _createCorrelationRow(corr, index) {
    const row = createElement('div', 'training-panel__correlation-row');
    row.dataset.index = String(index);

    const featureSelect = this._createFeatureSelect(corr.featureKey);
    featureSelect.addEventListener('change', () => {
      this.state.correlations[index].featureKey = featureSelect.value;
      this._persistCorrelations();
      this._renderCorrelations();
    });

    const targetSelect = document.createElement('select');
    PARTICLE_PARAMETER_TARGETS.forEach((target) => {
      const option = document.createElement('option');
      option.value = target.id;
      option.textContent = target.label;
      option.selected = target.id === corr.targetId;
      targetSelect.appendChild(option);
    });
    targetSelect.addEventListener('change', () => {
      this.state.correlations[index].targetId = targetSelect.value;
      this._persistCorrelations();
      this._renderCorrelations();
    });

    const strengthWrapper = createElement('div', 'training-panel__strength');
    const strengthLabel = createElement('span', '', formatPercent(corr.strength));
    const strengthInput = document.createElement('input');
    strengthInput.type = 'range';
    strengthInput.min = '0';
    strengthInput.max = '1';
    strengthInput.step = '0.01';
    strengthInput.value = corr.strength;
    strengthInput.addEventListener('input', () => {
      const value = clamp(Number(strengthInput.value), 0, 1);
      this.state.correlations[index].strength = value;
      strengthLabel.textContent = formatPercent(value);
      this._persistCorrelations();
    });
    strengthWrapper.append(strengthLabel, strengthInput);

    const polaritySelect = document.createElement('select');
    POLARITY_OPTIONS.forEach((option) => {
      const opt = document.createElement('option');
      opt.value = option.id;
      opt.textContent = option.label;
      opt.selected = corr.polarity === option.id;
      polaritySelect.appendChild(opt);
    });
    polaritySelect.addEventListener('change', () => {
      this.state.correlations[index].polarity = polaritySelect.value;
      this._persistCorrelations();
    });

    const removeBtn = this._createActionButton('Remove', 'ghost', () => this._handleRemoveCorrelation(index));
    removeBtn.classList.add('training-panel__btn--small');

    row.append(featureSelect, targetSelect, strengthWrapper, polaritySelect, removeBtn);
    return row;
  }

  _handleRemoveCorrelation(index) {
    if (index < 0 || index >= this.state.correlations.length) {
      return;
    }
    this.state.correlations.splice(index, 1);
    this._persistCorrelations();
    this._renderCorrelations();
  }

  _persistCorrelations() {
    const sanitized = sanitizeCorrelationList(this.state.correlations, this.featureKeys, this._getSanitizeOptions());
    this.state.correlations = sanitized;
    this.trainingManager.setCorrelations(sanitized);
  }

  _createFeatureSelect(selectedKey) {
    const select = document.createElement('select');
    if (this.featureKeys.length) {
      const audioGroup = document.createElement('optgroup');
      audioGroup.label = 'Audio Features';
      this.featureKeys.forEach((key) => {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = this.featureLabels[key] || key;
        if (key === selectedKey) {
          option.selected = true;
        }
        audioGroup.appendChild(option);
      });
      select.appendChild(audioGroup);
    }
    if (this.positionalFeatures.length) {
      const particleGroup = document.createElement('optgroup');
      particleGroup.label = 'Particle Parameters';
      this.positionalFeatures.forEach((feature) => {
        const option = document.createElement('option');
        option.value = feature.id;
        option.textContent = feature.label || feature.id;
        if (feature.id === selectedKey) {
          option.selected = true;
        }
        particleGroup.appendChild(option);
      });
      select.appendChild(particleGroup);
    }
    return select;
  }

  _getSanitizeOptions() {
    return { positionalFeatures: this.positionalFeatures };
  }

  async _handleStart() {
    this._clearError();
    try {
      await this.trainingManager.startTraining(this.state.trainingOptions);
    } catch (error) {
      this._showError(error?.message || 'Unable to start training.');
    }
  }

  async _handleFinetune() {
    this._clearError();
    if (!this.trainingManager) {
      this._showError('Training manager unavailable.');
      return;
    }
    if (!this.mlpModel) {
      this._showError('MLP Model unavailable.');
      return;
    }
    const button = this.buttons.finetune;
    if (button) {
      button.disabled = true;
      button.textContent = 'Finetuning...';
    }
    try {
      const snapshot = await this.mlpModel.exportSnapshot({
        label: 'Finetune base',
        correlations: this.state.correlations,
      });
      const weights = snapshot?.weights;
      if (!weights?.length) {
        throw new Error('Unable to capture current model weights.');
      }
      await this.trainingManager.startTraining({
        ...this.state.trainingOptions,
        baseWeights: weights,
      });
    } catch (error) {
      this._showError(error?.message || 'Unable to finetune model.');
    } finally {
      if (button) {
        button.textContent = 'Finetune';
        button.disabled = this.state.status === 'running';
      }
    }
  }

  async _handleRandomModel() {
    this._clearError();
    if (!this.mlpModel) {
      this._showError('MLP Model unavailable.');
      return;
    }
    if (this.state.status === 'running') {
      this._showError('Pause or abort training before randomizing the model.');
      return;
    }
    const button = this.buttons.random;
    if (button) {
      button.disabled = true;
      button.textContent = 'Randomizing...';
    }
    try {
      const config = this.mlpModel.getConfig?.();
      if (!config) {
        throw new Error('MLP configuration unavailable.');
      }
      const seed = Date.now();
      await this.mlpModel.rebuild({ seed });
      if (typeof this.trainingManager?.updateTrainingOptions === 'function') {
        this.trainingManager.updateTrainingOptions({ seed });
      }
      await this.mlpController?.syncModelDimensions?.();
      await this.mlpController?.runOnce?.();
      this.handleModelRandomized({ seed, reason: 'manual' });
    } catch (error) {
      this._showError(error?.message || 'Failed to randomize model.');
    } finally {
      if (button) {
        button.textContent = 'Random model';
        button.disabled = this.state.status === 'running';
      }
    }
  }

  _syncStatus(state = {}) {
    const status = state.status || this.trainingManager.getState?.().status || 'idle';
    const epoch = state.epoch ?? this.trainingManager.getState?.().epoch ?? 0;
    const epochs = state.epochs ?? this.trainingManager.getState?.().epochs ?? this.state.trainingOptions.epochs;
    const loss = state.loss ?? this.trainingManager.getState?.().loss ?? null;

    this.state.status = status;
    this.state.epoch = epoch;
    this.state.epochs = epochs;
    this.state.loss = loss;

    if (this.statusEl) {
      this.statusEl.textContent = statusLabels[status] || status;
    }
    const progress = epochs ? Math.min(1, epoch / epochs) : 0;
    if (this.progressEl?.firstElementChild) {
      this.progressEl.firstElementChild.style.transform = `scaleX(${progress})`;
      this.progressEl.setAttribute('aria-valuenow', progress);
    }
    if (this.lossEl) {
      this.lossEl.textContent = Number.isFinite(loss) ? `Loss: ${loss.toFixed(4)}` : '';
    }

    if (this.buttons.start) {
      this.buttons.start.disabled = status === 'running';
    }
    if (this.buttons.random) {
      this.buttons.random.disabled = status === 'running';
    }
    if (this.buttons.finetune) {
      this.buttons.finetune.disabled = status === 'running';
    }
    if (this.buttons.pause) {
      this.buttons.pause.disabled = status !== 'running';
    }
    if (this.buttons.resume) {
      this.buttons.resume.disabled = status !== 'paused';
    }
    if (this.buttons.abort) {
      this.buttons.abort.disabled = status === 'idle' || status === 'completed';
    }
  }

  _renderResult(metadata) {
    if (!this.resultEl) return;
    if (!metadata) {
      this.resultEl.textContent = 'No training run yet.';
      return;
    }
    const list = createElement('ul', 'training-panel__result-list');
    (metadata.correlations || []).forEach((item) => {
      const entry = createElement('li');
      const desired = formatPercent(item.strength);
      const achieved = formatPercent(Math.abs(item.achieved || 0));
      const targetLabel = this.targetLabels[item.targetId] || item.targetId;
      entry.textContent = `${this.featureLabels[item.featureKey] || item.featureKey} → ${targetLabel}: target ${desired}, achieved ${achieved} (${item.polarity === 'inverse' ? 'inverse' : 'direct'})`;
      list.appendChild(entry);
    });
    this.resultEl.innerHTML = '';
    const summary = createElement(
      'p',
      '',
      `${metadata.status === 'completed' ? 'Training completed' : 'Best checkpoint'} at epoch ${
        metadata.epochs
      } (loss ${metadata.loss.toFixed(4)}).`,
    );
    this.resultEl.append(summary, list);
  }

  async _handleExport() {
    this._clearError();
    if (!this.mlpModel) {
      this._showError('MLP Model unavailable.');
      return;
    }
    try {
      const snapshot = await this.mlpModel.exportSnapshot({
        label: this.state.exportName || 'Custom Model',
        notes: this.state.exportNotes || '',
        correlations: this.state.correlations,
        training: this.trainingManager.getState?.(),
      });
      const dataStr = JSON.stringify(snapshot, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      link.download = `${snapshot.metadata?.label || 'emergent-model'}-${timestamp}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      this._showError(error?.message || 'Failed to export model.');
    }
  }

  _handleImportClick() {
    this._clearError();
    this.fileInput?.click();
  }

  async _handleImport(event) {
    const [file] = event.target.files || [];
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      const snapshot = JSON.parse(text);
      await this.mlpModel.importSnapshot(snapshot);
      await this.mlpController?.runOnce?.();
      this.resultEl.textContent = `Loaded snapshot "${snapshot.metadata?.label || file.name}".`;
    } catch (error) {
      this._showError(error?.message || 'Failed to import snapshot.');
    } finally {
      event.target.value = '';
    }
  }

  handleModelRandomized({ seed, reason = 'manual', track = null } = {}) {
    if (!Number.isFinite(seed)) {
      return;
    }
    this.state.trainingOptions.seed = seed;
    if (this.resultEl) {
      const trackLabel = track?.title || track?.id || '';
      let prefix = 'Model randomized';
      if (reason === 'startup') {
        prefix = 'Session seeded a random model';
      } else if (reason === 'track') {
        prefix = trackLabel ? `Track "${trackLabel}" seeded a random model` : 'New track seeded a random model';
      } else if (reason === 'upload') {
        prefix = trackLabel ? `Upload "${trackLabel}" seeded a random model` : 'Uploaded track seeded a random model';
      }
      this.resultEl.textContent = `${prefix} (seed ${seed}).`;
    }
    this._clearError();
  }

  _showError(message) {
    if (!this.errorEl) return;
    this.errorEl.hidden = false;
    this.errorEl.textContent = message;
  }

  _clearError() {
    if (!this.errorEl) return;
    this.errorEl.hidden = true;
    this.errorEl.textContent = '';
  }
}
