import React from 'react';
import './SliderBlock.css'; // 确保引入了对应的样式文件
import HelpIcon from './HelpIcon';

const SliderBlock = ({ title, description, value, onChange, min = 0, max = 100, helpText }) => {
  return (
    <div className="slider-block">
      <label className="slider-title">
        {title}
        {helpText && (
          <span style={{ marginLeft: 6, verticalAlign: 'middle' }}>
            <HelpIcon text={helpText} />
          </span>
        )}
      </label>
      {description && (
        <p className="slider-description">{description}</p>
      )}
      <div className="slider-container">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="slider-input"
        />
        <span className="slider-value">{value}</span>
      </div>
    </div>
  );
};

export default SliderBlock;
