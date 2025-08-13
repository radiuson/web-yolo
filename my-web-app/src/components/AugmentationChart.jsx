import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from "chart.js";
import { Bar } from "react-chartjs-2";

// 注册组件（去掉 Legend）
ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

const COLORS = [
  "#4dc9f6", "#f67019", "#f53794", "#537bc4",
  "#acc236", "#166a8f", "#00a950", "#58595b", "#8549ba",
];

const AUGMENTATION_JA_LABELS = {
  original: "元画像",
  blur: "ぼかし",
  contrast: "コントラスト",
  rotate: "回転",
  brightness: "明るさ調整",
  noise: "ノイズ",
};

const AugmentationChart = ({ data }) => {
  if (!data || Object.keys(data).length === 0) {
    return <div>NULL</div>;
  }

    const total = Object.values(data).reduce((sum, v) => sum + v, 0);
    const keys = Object.keys(data);
    const categories = ["original", ...keys.filter((k) => k !== "original")]; // original 优先

    const datasets = categories.map((key, index) => {
    const isFirst = index === 0;
    const isLast = index === categories.length - 1;
    const borderRadius = {
        topLeft: isFirst ? 10 : 0,
        bottomLeft: isFirst ? 10 : 0,
        topRight: isLast ? 10 : 0,
        bottomRight: isLast ? 10 : 0,
    };

    return {
        label: AUGMENTATION_JA_LABELS[key] || key,
        data: [(data[key] / total * 100).toFixed(2)],
        backgroundColor: COLORS[index % COLORS.length],
        borderWidth: 0,
        borderRadius,      // ✅ 指定四个方向
        borderSkipped: false,
    };
    });

  const chartData = {
    labels: [""], // ✅ 空标签：不显示左边 label
    datasets,
  };
    const options = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false, // ✅ 允许自适应容器
    scales: {
        x: {
        stacked: true,
        max: 100,
        ticks: {
            display: false,
        },
        grid: {
            display: false,
            drawBorder: false,
        },
        },
        y: {
        stacked: true,
        ticks: {
            display: false,
        },
        grid: {
            display: false,
            drawBorder: false,
        },
        },
    },
    plugins: {
        tooltip: {
        callbacks: {
            label: (ctx) => {
            const engKey = Object.keys(data).find(
                (k) =>
                AUGMENTATION_JA_LABELS[k] === ctx.dataset.label ||
                k === ctx.dataset.label
            );
            const label = ctx.dataset.label;
            const percent = ctx.dataset.data[0];
            const count = data[engKey];
            return `${label}: ${count} 枚 (${percent}%)`;
            },
        },
        },
        legend: {
        display: false,
        },
    },
    };



    return (
    <div style={{ maxWidth: "600px", margin: "auto" }}>
        <h3 style={{ textAlign: "center", marginBottom: "10px" }}>
        データ拡張の割合
        </h3>
        
        {/* ✅ 只限制图表高度 */}
        <div style={{ height: "40px" }}>
        <Bar data={chartData} options={options} />
        </div>
    </div>
    );

};

export default AugmentationChart;
