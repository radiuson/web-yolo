import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LineController,
  Legend,
} from 'chart.js';
import "./ResultsChart.css";
import HelpIcon from './HelpIcon';
ChartJS.register(LineElement, PointElement, LineController, Legend);

const BASE_API = "http://www.ihpc.se.ritsumei.ac.jp/detect";
const apiUrl = `${BASE_API}/results_csv`;

const helpTexts = {
  accuracy: "正確率はモデルが正しく予測した割合を示します。高いほど良い性能を意味します。",
  loss: "損失はモデルの誤差を示し、学習の進み具合を表します。小さいほど良いです。",
};
const ResultsChart = ({ csvPath }) => {
  const [chartData, setChartData] = useState(null);
  const [lossData, setLossData] = useState(null); // 新增：存放loss数据

  // 图表配置
  const options = {
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          boxWidth: 20,
          padding: 15,
          color: 'black'
        },
      },
    },
  };

  // Loss图表配置
  const lossOptions = {
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: {
          boxWidth: 20,
          padding: 15,
          color: 'black'
        },
      },
    },
  };

  useEffect(() => {
    let intervalId;

    const fetchData = () => {
      console.log('开始请求数据');
      fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: csvPath }),
      })
        .then(res => {
          if (!res.ok) {
            if (res.status === 404) {
              console.log('文件还没有生成');
              return { error: 'not_found' };
            }
            throw new Error(`HTTP error! status: ${res.status}`);
          }
          return res.json();
        })
        .then(data => {
          if (data.error) {
            if (data.error === 'not_found') {
              console.log('文件还没有生成');
              return;
            }
            console.error('返回错误:', data.error);
            return;
          }
          const epochs = data.map(row => parseInt(row['epoch']));
          const mAP50 = data.map(row => parseFloat(row['metrics/mAP50(B)']));
          const mAP50_95 = data.map(row => parseFloat(row['metrics/mAP50-95(B)']));
          
          // 提取loss
          const trainBoxLoss = data.map(row => parseFloat(row['train/box_loss']));
          const trainClsLoss = data.map(row => parseFloat(row['train/cls_loss']));
          const trainDflLoss = data.map(row => parseFloat(row['train/dfl_loss']));

          // 设置主图表数据
          setChartData({
            labels: epochs.map(e => e.toString()), // 转成字符串
            datasets: [
              {
                label: 'mAP50',
                data: mAP50,
                borderColor: 'blue',
                fill: false,
              },
              {
                label: 'mAP50-95',
                data: mAP50_95,
                borderColor: 'red',
                fill: false,
              },
            ],
          });

          // 设置loss图表数据
          setLossData({
            labels: epochs.map(e => e.toString()),
            datasets: [
              {
                label: 'box_loss',
                data: trainBoxLoss,
                borderColor: 'orange',
                fill: false,
              },
              {
                label: 'cls_loss',
                data: trainClsLoss,
                borderColor: 'green',
                fill: false,
              },
              {
                label: 'dfl_loss',
                data: trainDflLoss,
                borderColor: 'purple',
                fill: false,
              },
            ],
          });
        })
        .catch(error => {
          console.error('请求错误:', error);
        });
    };

    fetchData();
    intervalId = setInterval(fetchData, 3000);

    return () => clearInterval(intervalId);
  }, [apiUrl, csvPath]);

  if (!chartData || !lossData) return <div>Loading...</div>;

  return (
      <div className="charts-wrapper">
        <div className="chart-container">
        <h3>正確率</h3>
          <Line data={chartData} options={options} />
        </div>
        <div className="chart-container">
          <h3>損失</h3>
          <Line data={lossData} options={lossOptions} />
        </div>
      </div>
  );

};

export default ResultsChart;
