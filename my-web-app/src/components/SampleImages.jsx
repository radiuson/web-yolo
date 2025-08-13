import React, { useState, useEffect } from "react";
import "./SampleImages.css"; // 你可以把样式写在这里

const SampleImages = ({ images, onSelect }) => {
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    if (images && images.length > 0) {
      setSelectedIdx(0);
      if (onSelect) {
        onSelect(`data:image/jpeg;base64,${images[0].base64}`);
      }
    }
  }, [images, onSelect]);

const handleImageClick = (idx) => {
  if (selectedIdx === idx) {
    setShowModal(true); // 二次点击放大
  } else {
    setSelectedIdx(idx);
    if (onSelect) {
      onSelect(`data:image/jpeg;base64,${images[idx].base64}`); // ✅ 传给父组件
    }
  }
};

  return (
    <>
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
        {images.map((img, idx) => (
          <img
            key={idx}
            src={`data:image/jpeg;base64,${img.base64}`}
            alt={img.filename || `sample-${idx}`}
            onClick={() => handleImageClick(idx)}
            style={{
              width: "80px",
              height: "80px",
              objectFit: "cover",
              borderRadius: "8px",
              border: selectedIdx === idx ? "3px solid #2196f3" : "1px solid #ccc",
              opacity: selectedIdx === idx ? 1 : 0.8,
              cursor: "pointer",
            }}
          />
        ))}
      </div>

      {/* 弹出 Modal 显示大图 */}
      {showModal && selectedIdx !== null && (
        <div
          className="image-modal-overlay"
          onClick={() => setShowModal(false)}
        >
          <img
            src={`data:image/jpeg;base64,${images[selectedIdx].base64}`}
            alt="enlarged"
            className="image-modal-content"
          />
        </div>
      )}
    </>
  );
};

export default SampleImages;
