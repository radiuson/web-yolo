import React, { useState } from "react";
import AugmentationModal from "./AugmentationModal";
import "./AugmentationButton.css";

const AugmentationButton = ({
  type,
  label,
  sampleImageBase64,
  setAugmentationSettings,
}) => {
  const [showModal, setShowModal] = useState(false);
  const [isConfirmed, setIsConfirmed] = useState(false);

  return (
    <>
      <button
        className={`augmentation-btn ${isConfirmed ? "confirmed" : ""}`}
        onClick={() => setShowModal(true)}
      >
        {label}
      </button>

      {showModal && (
        <AugmentationModal
          type={type}
          label={label}
          sampleImageBase64={sampleImageBase64}
          onClose={() => setShowModal(false)}
          onConfirm={(setting) => {
            if (setting === null || setting.scale === 0) {
              // 用户取消增强（倍率 = 0 或返回 null）
              setIsConfirmed(false);
              setAugmentationSettings?.((prev) => {
                const updated = { ...prev };
                delete updated[type];
                return updated;
              });
            } else {
              // 用户启用增强
              setIsConfirmed(true);
              setAugmentationSettings?.((prev) => ({
                ...prev,
                [setting.type]: setting,
              }));
            }
            setShowModal(false);
          }}
        />
      )}
    </>
  );
};

export default AugmentationButton;
