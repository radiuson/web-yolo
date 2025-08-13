import React from "react";

const ActionButtons = ({ imageUri, onClear, onSubmit }) => {
  const hasImage = Boolean(imageUri);

  return (
    <div className="flex items-center gap-3 mt-4">
      {/* 删除按钮 */}
      <button
        onClick={onClear}
        disabled={!hasImage}
        className={`px-4 py-2 rounded border ${
          hasImage
            ? "bg-red-500 text-white border-red-500 hover:bg-red-600"
            : "bg-gray-100 text-red-500 border-red-500 cursor-not-allowed"
        }`}
      >
        画像削除
      </button>

      {/* 送信按钮 */}
      <button
        onClick={onSubmit}
        disabled={!hasImage}
        className={`flex-1 px-4 py-2 rounded ${
          hasImage
            ? "bg-blue-500 text-white hover:bg-blue-600"
            : "bg-slate-400 text-white cursor-not-allowed"
        }`}
      >
        送信
      </button>
    </div>
  );
};

export default ActionButtons;
