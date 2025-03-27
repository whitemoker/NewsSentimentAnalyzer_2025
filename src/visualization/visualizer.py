import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional
import json

class SummaryVisualizer:
    """摘要可视化工具，用于生成Rouge指标和情感一致性的可视化图表"""
    
    def __init__(self, output_dir: str = "visualization_results"):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_rouge_scores(self, results: Dict[str, Dict[str, float]], output_filename: str = "rouge_scores.png"):
        """
        绘制Rouge指标对比图
        
        Args:
            results: 包含模型名称和Rouge指标的字典
            output_filename: 输出文件名
        """
        data = []
        for model_name, scores in results.items():
            for metric, value in scores.items():
                data.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Score": value
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x="Metric", y="Score", hue="Model", data=df)
        plt.title("Rouge Scores Comparison", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.legend(title="Model", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def plot_sentiment_consistency(self, results: Dict[str, Dict[str, float]], output_filename: str = "sentiment_consistency.png"):
        """
        绘制情感一致性对比图
        
        Args:
            results: 包含模型名称和情感一致性指标的字典
            output_filename: 输出文件名
        """
        models = list(results.keys())
        label_consistency = [results[model].get("label_consistency", 0) for model in models]
        score_consistency = [results[model].get("score_consistency", 0) for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, label_consistency, width, label='标签一致性')
        rects2 = ax.bar(x + width/2, score_consistency, width, label='分数一致性')
        
        ax.set_title("情感一致性对比", fontsize=16)
        ax.set_xlabel("模型", fontsize=14)
        ax.set_ylabel("一致性分数", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def create_metrics_heatmap(self, results: Dict[str, Dict[str, Any]], output_filename: str = "metrics_heatmap.png"):
        """
        创建指标热力图
        
        Args:
            results: 包含模型名称和各项指标的字典
            output_filename: 输出文件名
        """
        metrics_data = {}
        for model_name, model_results in results.items():
            model_metrics = {}
            
            if "rouge_scores" in model_results:
                rouge = model_results["rouge_scores"]
                for metric in ["rouge-1", "rouge-2", "rouge-l"]:
                    if metric in rouge:
                        model_metrics[f"{metric}-F1"] = rouge[metric].get("f", 0)
                        model_metrics[f"{metric}-Precision"] = rouge[metric].get("p", 0)
                        model_metrics[f"{metric}-Recall"] = rouge[metric].get("r", 0)
            
            if "sentiment_consistency" in model_results:
                sentiment = model_results["sentiment_consistency"]
                model_metrics["标签一致性"] = sentiment.get("label_consistency", 0)
                model_metrics["分数一致性"] = sentiment.get("score_consistency", 0)
            
            metrics_data[model_name] = model_metrics
        
        df = pd.DataFrame(metrics_data).T
        
        plt.figure(figsize=(14, len(results) * 1.2))
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
        plt.title("模型性能指标热力图", fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def create_comparison_dashboard(self, results_dir: str, output_filename: str = "comparison_dashboard.html"):
        """
        创建模型对比仪表板
        
        Args:
            results_dir: 包含评估结果的目录
            output_filename: 输出HTML文件名
        """
        model_results = {}
        for filename in os.listdir(results_dir):
            if filename.endswith("_results.json"):
                model_name = filename.replace("_results.json", "")
                with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                    model_results[model_name] = json.load(f)
        
        if not model_results:
            print("未找到评估结果文件")
            return None
        
        rouge_results = {}
        for model_name, results in model_results.items():
            if "rouge_scores" in results:
                rouge_scores = {}
                for metric in ["rouge-1", "rouge-2", "rouge-l"]:
                    if metric in results["rouge_scores"]:
                        rouge_scores[f"{metric}-F1"] = results["rouge_scores"][metric].get("f", 0)
                rouge_results[model_name] = rouge_scores
        
        rouge_chart_path = self.plot_rouge_scores(rouge_results, "dashboard_rouge.png")
        
        sentiment_results = {}
        for model_name, results in model_results.items():
            if "sentiment_consistency" in results:
                sentiment_results[model_name] = results["sentiment_consistency"]
        
        sentiment_chart_path = self.plot_sentiment_consistency(sentiment_results, "dashboard_sentiment.png")
        
        heatmap_path = self.create_metrics_heatmap(model_results, "dashboard_heatmap.png")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>中文新闻摘要模型对比仪表板</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .chart-container {{
                    margin: 30px 0;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>中文新闻摘要模型对比仪表板</h1>
                
                <h2>Rouge指标对比</h2>
                <div class="chart-container">
                    <img src="dashboard_rouge.png" alt="Rouge Scores Comparison">
                </div>
                
                <h2>情感一致性对比</h2>
                <div class="chart-container">
                    <img src="dashboard_sentiment.png" alt="Sentiment Consistency Comparison">
                </div>
                
                <h2>性能指标热力图</h2>
                <div class="chart-container">
                    <img src="dashboard_heatmap.png" alt="Metrics Heatmap">
                </div>
                
                <h2>模型详细指标</h2>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>Rouge-1-F1</th>
                        <th>Rouge-2-F1</th>
                        <th>Rouge-L-F1</th>
                        <th>情感标签一致性</th>
                        <th>情感分数一致性</th>
                    </tr>
        """
        
        for model_name, results in model_results.items():
            rouge_1_f = results.get("rouge_scores", {}).get("rouge-1", {}).get("f", 0)
            rouge_2_f = results.get("rouge_scores", {}).get("rouge-2", {}).get("f", 0)
            rouge_l_f = results.get("rouge_scores", {}).get("rouge-l", {}).get("f", 0)
            
            label_consistency = results.get("sentiment_consistency", {}).get("label_consistency", 0)
            score_consistency = results.get("sentiment_consistency", {}).get("score_consistency", 0)
            
            html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{rouge_1_f:.4f}</td>
                        <td>{rouge_2_f:.4f}</td>
                        <td>{rouge_l_f:.4f}</td>
                        <td>{label_consistency:.4f}</td>
                        <td>{score_consistency:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
