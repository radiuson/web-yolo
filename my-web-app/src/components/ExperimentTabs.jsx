import React, { useState } from 'react';
import "./ExperimentTabs.css";
import ResultsChart from "../components/ResultsChart";
const ExperimentTabs = ({ experiments, jobId, trainingInProgress, activeIndex, setActiveIndex, trainingExperimentPath }) => {

  // 解析时间戳
    const parseTime = (exp) => {
        const parts = exp.split('_');
        const dateStr = parts[1];
        const timeStr = parts[2];
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const hours = timeStr.substring(0, 2);
        const minutes = timeStr.substring(2, 4);
        const seconds = timeStr.substring(4, 6);
        const dateObj = new Date(`${year}-${month}-${day}T${hours}:${minutes}:${seconds}`);
        return dateObj.getTime();
    };

  // 按时间倒序排序（最新在左）
  const sortedExperiments = [...experiments].sort((a, b) => parseTime(b) - parseTime(a));
  // 只取前10个
  const topExperiments = sortedExperiments.slice(0, 10);

  return (
    <div className="full-width-container">
      {/* 标签栏 */}
        <div className="tabs-container">
        {/* 进行中tab */}
        <div
            className={`tab-item ${activeIndex === 0 ? 'active' : ''}`}
            onClick={() => setActiveIndex(0)}
        >
            + 進行中
        </div>
        {/* 其他实验标签 */}
        {topExperiments.map((exp, index) => {
            const displayIndex = index + 1; // 实验索引从1开始
            const parts = exp.split('_');
            const timeStr = parts[2];
            const hours = timeStr.substring(0, 2);
            const minutes = timeStr.substring(2, 4);
            const displayTime = `${hours}:${minutes}`;

            return (
            <div
                key={exp}
                className={`tab-item ${activeIndex === displayIndex ? 'active' : ''}`}
                onClick={() => setActiveIndex(displayIndex)}
            >
                {`#${displayIndex} ${displayTime}`}
            </div>
            );
        })}
        </div>

      {/* 内容区域 */}
      <div style={{ padding: '10px' }}>
        {activeIndex === 0 && trainingInProgress && trainingExperimentPath ? (
            <ResultsChart
            key={activeIndex}
            csvPath={trainingExperimentPath}
            />
        ) : (
          // 正常显示实验结果
          activeIndex !== 0 && experiments.length > 0 && (
            <ResultsChart
              key={activeIndex}
              csvPath={`job_${jobId}/results/${sortedExperiments[activeIndex - 1]}/results.csv`}
            />
          )
        )}
      </div>
    </div>
  );
};

export default ExperimentTabs;
