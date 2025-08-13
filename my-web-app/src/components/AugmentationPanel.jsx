import React from "react";
import AugmentationButton from "./AugmentationButton";
import "./AugmentationPanel.css";
import HelpIcon from "./HelpIcon";

const AugmentationPanel = ({
  sampleImageBase64,
  augmentationSettings,
  setAugmentationSettings,
}) => {
  return (
    <div className="augmentation-container">
      <h1 className="augmentation-title">データ拡張
        <HelpIcon text="データ拡張は、モデルの汎用性を高めるために画像に様々な変換を加える技術です。" />
      </h1>
      <div className="augmentation-buttons">
        <AugmentationButton
          type="blur"
          label="ぼかし"
          sampleImageBase64={sampleImageBase64}
          augmentationSettings={augmentationSettings}
          setAugmentationSettings={setAugmentationSettings}
        />
        <AugmentationButton
          type="rotate"
          label="回転"
          sampleImageBase64={sampleImageBase64}
          augmentationSettings={augmentationSettings}
          setAugmentationSettings={setAugmentationSettings}
        />
        <AugmentationButton
          type="brightness"
          label="明るさ調整"
          sampleImageBase64={sampleImageBase64}
          augmentationSettings={augmentationSettings}
          setAugmentationSettings={setAugmentationSettings}
        />
        <AugmentationButton
          type="contrast"
          label="コントラスト"
          sampleImageBase64={sampleImageBase64}
          augmentationSettings={augmentationSettings}
          setAugmentationSettings={setAugmentationSettings}
        />
        <AugmentationButton
          type="noise"
          label="ノイズ追加"
          sampleImageBase64={sampleImageBase64}
          augmentationSettings={augmentationSettings}
          setAugmentationSettings={setAugmentationSettings}
        />
      </div>
    </div>
  );
};

export default AugmentationPanel;
