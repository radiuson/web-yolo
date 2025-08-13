import React, { useEffect } from 'react';
import "./TrainingParameters.css"
import AdvancedOption from './AdvancedOption';
import HelpIcon from './HelpIcon';

const helpTexts = {
  epoch: "トレーニングの繰り返し回数です。多いほど学習が進みますが、時間がかかります。",
  learningRate: "モデルの重みを更新する際のステップサイズ。大きすぎると不安定、小さすぎると学習が遅くなります。",
  modelType: "使用するYOLOモデルの種類を選択します。軽量モデルから高精度モデルまであります。",
  usePretrained: "事前学習済みモデルを使うかどうか。使うと学習が速くなり精度も上がることがあります。",
  batch: "一度に処理する画像の枚数。大きいほど学習が安定しますがメモリを多く使います。",
  weightDecay: "過学習を防ぐための正則化パラメータです。通常は小さい値を設定します。",
  optimizer: "モデルの重みを更新するアルゴリズムを選択します。AdamWやSGDがあります。",
};

const TrainingParameters = ({
  selectedEpoch,
  setSelectedEpoch,
  optimizer,
  setOptimizer,
  learningRate,
  setLearningRate,
  batch,
  setBatch,
  weightDecay,
  setWeightDecay,
  isTraining,
  handleStartTraining,
  handleCancelTraining,  
  modelType,
  setModelType,
  usePretrained,
  setUsePretrained,
}) => {
  const learningRates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
  const currentIndex = learningRates.indexOf(learningRate);
  const sliderIndex = currentIndex === -1 ? 3 : currentIndex; // 默认1e-4
  const epochMax = usePretrained ? 30 : 200;
  useEffect(() => {
    if (selectedEpoch > epochMax) {
      setSelectedEpoch(epochMax); // 自动纠正超出范围
    }
  }, [epochMax, selectedEpoch, setSelectedEpoch]);
  return (
    <div className="parameter-box">
      <div className="parameter-item">
        <label htmlFor="epochRange">Epoch: {selectedEpoch}
          <HelpIcon text={helpTexts.epoch} />
        </label>
        <input
          type="range"
          id="epochRange"
          min={5}
          max={epochMax}
          step={1}
          value={selectedEpoch}
          onChange={(e) => setSelectedEpoch(parseInt(e.target.value))}
          disabled={isTraining}
        />
      </div>

      <div className="parameter-item">
        <label htmlFor="lrRange">初期学習率: {learningRates[sliderIndex].toExponential()}
          <HelpIcon text={helpTexts.learningRate} />
        </label>
        <input
          id="lrRange"
          type="range"
          min={0}
          max={learningRates.length - 1}
          step={1}
          value={sliderIndex}
          onChange={(e) => setLearningRate(learningRates[parseInt(e.target.value)])}
          disabled={isTraining}
        />
      </div>

      <AdvancedOption title="詳細設定">
        <div>
            <div className="parameter-item">
                <label htmlFor="modelSelect">YOLOモデル選択:
                  <HelpIcon text={helpTexts.modelType} />
                </label>
                <select
                    id="modelSelect"
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                    disabled={isTraining}
                >
                    <optgroup label="YOLOv5">
                    <option value="yolov5s">YOLOv5s</option>
                    </optgroup>
                    <optgroup label="YOLOv8">
                    <option value="yolov8n">YOLOv8n</option>
                    <option value="yolov8s">YOLOv8s</option>
                    </optgroup>
                </select>
            </div>
            <div className="parameter-item">
            <label htmlFor="pretrainedSelect">事前学習済みを使用するか:
              <HelpIcon text={helpTexts.usePretrained} />
            </label>
            <select
                id="pretrainedSelect"
                value={usePretrained ? 'yes' : 'no'}
                onChange={(e) => setUsePretrained(e.target.value === 'yes')}
                disabled={isTraining}
            >
                <option value="yes">はい</option>
                <option value="no">いいえ</option>
            </select>
            </div>


            <div className="parameter-item">
              <label htmlFor="batchSelect">
                Batch Size:
                <HelpIcon text={helpTexts.batch} />
              </label>
              <select
                id="batchSelect"
                value={batch}
                onChange={(e) => setBatch(parseInt(e.target.value))}
                disabled={isTraining}
              >
                {[1, 2, 4, 8, 16, 32].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </div>

            <div className="parameter-item">
                <label htmlFor="weightDecayInput">Weight Decay:
                  <HelpIcon text={helpTexts.weightDecay} />
                </label>
                <input
                id="weightDecayInput"
                type="number"
                min={0}
                step={0.0001}
                value={weightDecay}
                onChange={(e) => setWeightDecay(parseFloat(e.target.value) || 0)}
                disabled={isTraining}
                />
            </div>

            <div className="parameter-item">
                <label htmlFor="optimizerSelect">Optimizer:
                  <HelpIcon text={helpTexts.optimizer} />
                </label>
                <select
                id="optimizerSelect"
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
                disabled={isTraining}
                >
                <option value="AdamW">AdamW</option>
                <option value="SGD">SGD</option>
                </select>
            </div>
        </div>
      </AdvancedOption>

      {/* 按钮组 */}
      <div className="button-group" style={{ marginTop: 20 }}>
        <button onClick={handleStartTraining} disabled={isTraining}>
          {isTraining ? "トレーニング中…" : "トレーニング開始"}
        </button>

        <button onClick={handleCancelTraining} disabled={!isTraining} style={{ marginLeft: 10 }}>
          キャンセル
        </button>
      </div>
    </div>
  );
};

export default TrainingParameters;
