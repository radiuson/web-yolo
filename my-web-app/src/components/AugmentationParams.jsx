// AugmentationParams.jsx
import React from "react";

const AugmentationParams = ({ type, value, onChange }) => {
  switch (type) {
    case "blur":
      return (
        <>
          <label>
            ぼかし強度: {value}
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
            />
          </label>
        </>
      );

    case "rotate":
      return (
        <>
          <label>
            回転角度: {value}°
            <input
              type="range"
              min={-180}
              max={180}
              step={1}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
            />
          </label>
        </>
      );

    case "brightness":
      return (
        <>
          <label>
            明るさ調整: {value}%
            <input
              type="range"
              min={50}
              max={150}
              step={1}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
            />
          </label>
        </>
      );

    case "contrast":
      return (
        <>
          <label>
            コントラスト: {value}%
            <input
              type="range"
              min={50}
              max={200}
              step={1}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
            />
          </label>
        </>
      );

    case "noise":
      return (
        <>
          <label>
            ノイズ量: {value}%
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
            />
          </label>
        </>
      );

    default:
      return <p>この拡張方法にはパラメータはありません。</p>;
  }
};

export default AugmentationParams;
