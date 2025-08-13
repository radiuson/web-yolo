import React, { useState } from 'react';
import './ImagePreview.css';

const ImagePreview = ({ imageUri }) => {
  const [modalOpen, setModalOpen] = useState(false);

  if (!imageUri) {
    return <div style={{ color: '#888', textAlign: 'center' }}>画像はない</div>;
  }

  return (
    <>
      {/* 缩略图 */}
      <img
        src={imageUri}
        alt="preview"
        onClick={() => setModalOpen(true)}
        className="image-preview"
      />

      {/* 放大图弹窗 */}
      {modalOpen && (
        <div
          className="modal-overlay"
          onClick={() => setModalOpen(false)}
        >
          <img
            src={imageUri}
            alt="fullscreen"
            onClick={(e) => e.stopPropagation()} // 防止点图关闭
          />
        </div>
      )}
    </>
  );
};

export default ImagePreview;
