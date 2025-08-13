import React, { useState } from "react";
import AugmentationParams from "./AugmentationParams";
import "./AugmentationModal.css";
import HelpIcon from "./HelpIcon";

const helpTexts = {
  blur: "画像をぼかすデータ拡張です。画像の輪郭を柔らかくし、ノイズを減らす効果があります。",
  rotate: "画像を回転させるデータ拡張です。モデルの回転に対する頑健性を高めます。",
  brightness: "画像の明るさを調整するデータ拡張です。明るさの変化に対応できるようにします。",
  contrast: "画像のコントラストを調整するデータ拡張です。画像の濃淡差を強調または弱めます。",
  noise: "画像にノイズを加えるデータ拡張です。ノイズに対する耐性を向上させます。",
};


const getDefaultValue = (type) => {
  switch (type) {
    case "blur":
      return 3;
    case "rotate":
      return 15;
    case "brightness":
      return 100;
    case "contrast":
      return 100;
    case "noise":
      return 10;
    default:
      return 0;
  }
};

const handleApply = async () => {
  const res = await fetch("/augment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: sampleImageBase64,
      type,
      param: Number(sliderValue),
    }),
  });
  const result = await res.json();
  setPreviewImage(result.image); // 更新预览图
};

const AugmentationModal = ({ type, label, sampleImageBase64, onClose, onConfirm }) => {

    const [paramValue, setParamValue] = useState(getDefaultValue(type));
    const [previewImage, setPreviewImage] = useState(sampleImageBase64 || "");
    const [scaleFactor, setScaleFactor] = useState(1); 
  
  return (
    <div className="augmentation-modal-overlay">
      <div className="augmentation-modal-content">
        <h3>{label} 
           <HelpIcon text={helpTexts[type] || "データ拡張の説明はありません。"} />
          設定</h3>

        {sampleImageBase64 ? (
          <img src={previewImage} alt="preview" className="augmentation-preview-img" />
        ) : (
          <p>画像が選択されていません</p>
        )}

        <div className="augmentation-param-box">
          <AugmentationParams
            type={type}
            value={paramValue}
            onChange={setParamValue}
          />
        </div>

        <div className="augmentation-scale-box">
          <label>
            {scaleFactor === 0 ? "使用しない" : `拡張倍率: ${scaleFactor} 倍`}
            <input
              type="range"
              min={0}
              max={3}
              step={1}
              value={scaleFactor}
              onChange={(e) => setScaleFactor(Number(e.target.value))}
            />

          </label>
        </div>



        <div className="augmentation-modal-buttons">
          <button onClick={onClose}>キャンセル</button>
          <button
            onClick={async () => {
                try {
                const response = await fetch("http://www.ihpc.se.ritsumei.ac.jp/detect/augment_preview", {
                    method: "POST",
                    headers: {
                    "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                    type,
                    param: paramValue,
                    image: sampleImageBase64,
                    }),
                });

                const data = await response.json();

                if (data.image) {
                    // ✅ 更新预览图
                    setPreviewImage(data.image);
                } else {
                    alert("拡張失敗: " + data.error);
                }
                } catch (err) {
                console.error("拡張失敗", err);
                alert("拡張失敗");
                }
            }}
            >
            適用
          </button>

          <button
            onClick={() => {
              console.log("最终传出的 setting：", scaleFactor, paramValue);
              if (scaleFactor === 0) {
                // 返回 null 表示未使用
                onConfirm?.(null);
              } else {
                onConfirm?.({
                  type,
                  param: paramValue,
                  scale: scaleFactor,
                });
              }
              onClose();
            }}
          >
            確定
          </button>


        </div>
      </div>
    </div>
  );
};

export default AugmentationModal;
