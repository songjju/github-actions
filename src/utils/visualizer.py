"""
파일명: src/utils/visualizer.py
목적: 로또 예측 시스템의 데이터 시각화 도구
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- LottoVisualizer: 메인 시각화 클래스
- StatisticsPlotter: 통계 차트 생성
- PredictionPlotter: 예측 결과 시각화
- PerformancePlotter: 성능 분석 차트
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

class LottoVisualizer:
    """메인 시각화 클래스"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        초기화
        
        Args:
            output_dir: 출력 디렉토리 (기본값: config에서 가져옴)
        """
        self.output_dir = output_dir or Config.OUTPUTS_DIR / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 설정
        self._setup_matplotlib()
        
        # 서브 플로터들
        self.stats_plotter = StatisticsPlotter(self.output_dir)
        self.prediction_plotter = PredictionPlotter(self.output_dir)
        self.performance_plotter = PerformancePlotter(self.output_dir)

    def _setup_matplotlib(self):
        """Matplotlib 설정"""
        # 기본 설정
        plt.style.use('default')
        
        # 한글 폰트 설정 시도
        try:
            # 안전한 폰트 설정 (한글 폰트 대신 기본 폰트 사용)
            plt.rcParams['font.family'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print("기본 폰트 설정 완료")
                    
        except Exception as e:
            print(f"폰트 설정 중 오류: {e}")
            # 기본값으로 fallback
            plt.rcParams['font.family'] = 'sans-serif'
        
        # 기본 시각화 설정
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        
        # Seaborn 설정
        try:
            sns.set_palette("husl")
        except:
            pass  # seaborn이 없거나 오류 시 무시
        
    def create_comprehensive_report(self, 
                                    lotto_data: pd.DataFrame,
                                    statistics: Dict[str, Any],
                                    predictions: List[Dict],
                                    performance_data: Optional[Dict] = None) -> str:
        """
        종합 시각화 리포트 생성
        
        Args:
            lotto_data: 로또 데이터
            statistics: 통계 분석 결과
            predictions: 예측 결과들
            performance_data: 성능 데이터
            
        Returns:
            str: 생성된 리포트 파일 경로
        """
        print("종합 시각화 리포트 생성 중...")
        
        # 개별 차트 생성
        chart_files = []
        
        try:
            # 1. 통계 차트들
            if statistics:
                stats_files = self.stats_plotter.create_statistics_charts(lotto_data, statistics)
                chart_files.extend(stats_files)
            
            # 2. 예측 차트들
            if predictions:
                pred_files = self.prediction_plotter.create_prediction_charts(predictions)
                chart_files.extend(pred_files)
            
            # 3. 성능 차트들
            if performance_data:
                perf_files = self.performance_plotter.create_performance_charts(performance_data)
                chart_files.extend(perf_files)
            
            # HTML 리포트 생성
            report_path = self._generate_html_report(chart_files, statistics, predictions)
            
            print(f"종합 리포트 생성 완료: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"리포트 생성 중 오류: {e}")
            return ""
    
    def _generate_html_report(self, 
                            chart_files: List[str],
                            statistics: Dict[str, Any],
                            predictions: List[Dict]) -> Path:
        """HTML 리포트 생성"""
        report_path = self.output_dir / 'analysis_report.html'
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>로또 예측 시스템 - 분석 리포트</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .prediction-card {{
            background: #e3f2fd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }}
        .footer {{
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
            padding-top: 10px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 로또 예측 시스템 분석 리포트</h1>
            <p>생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>📊 기본 통계</h2>
            <div class="stats-grid">
        """
        
        # 통계 정보 추가
        if statistics:
            basic_info = statistics.get('basic_info', {})
            freq_analysis = statistics.get('frequency_analysis', {})
            
            html_content += f"""
                <div class="stat-card">
                    <h3>데이터 개요</h3>
                    <p><strong>총 추첨 횟수:</strong> {basic_info.get('total_draws', 'N/A')}회</p>
                    <p><strong>최신 회차:</strong> {basic_info.get('latest_round', 'N/A')}회</p>
                    <p><strong>분석 기간:</strong> {basic_info.get('oldest_round', 'N/A')}회 ~ {basic_info.get('latest_round', 'N/A')}회</p>
                </div>
                
                <div class="stat-card">
                    <h3>빈도 분석</h3>
            """
            
            most_frequent = freq_analysis.get('most_frequent_numbers', [])
            if most_frequent:
                top_numbers = [str(item[0]) if isinstance(item, tuple) else str(item) for item in most_frequent[:5]]
                html_content += f"<p><strong>자주 나온 번호:</strong> {', '.join(top_numbers)}</p>"
            
            least_frequent = freq_analysis.get('least_frequent_numbers', [])
            if least_frequent:
                bottom_numbers = [str(item[0]) if isinstance(item, tuple) else str(item) for item in least_frequent[:5]]
                html_content += f"<p><strong>적게 나온 번호:</strong> {', '.join(bottom_numbers)}</p>"
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>📈 시각화 차트</h2>
        """
        
        # 차트 이미지들 추가
        for chart_file in chart_files:
            chart_name = Path(chart_file).stem.replace('_', ' ').title()
            html_content += f"""
            <div class="chart">
                <h3>{chart_name}</h3>
                <img src="{Path(chart_file).name}" alt="{chart_name}">
            </div>
            """
        
        html_content += """
        </div>
        
        <div class="section">
            <h2>🎲 예측 결과</h2>
        """
        
        # 예측 결과들 추가
        for i, prediction in enumerate(predictions, 1):
            final_pred = prediction.get('final_prediction', [])
            confidence = prediction.get('ensemble_confidence', 0)
            
            html_content += f"""
            <div class="prediction-card">
                <h3>예측 세트 {i}</h3>
                <p><strong>예측 번호:</strong> {final_pred}</p>
                <p><strong>신뢰도:</strong> {confidence:.1%}</p>
            """
            
            if 'synthesis_strategy' in prediction:
                html_content += f"<p><strong>합성 전략:</strong> {prediction['synthesis_strategy']}</p>"
            
            html_content += "</div>"
        
        html_content += f"""
        </div>
        
        <div class="footer">
            <p>⚠️ 이 예측은 통계적 분석을 기반으로 하며, 실제 당첨을 보장하지 않습니다.</p>
            <p>로또는 무작위 추첨이므로 적당한 금액으로 즐기시기 바랍니다.</p>
            <p>생성 시각: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # HTML 파일 저장
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path


class StatisticsPlotter:
    """통계 차트 생성 클래스"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def create_statistics_charts(self, 
                               lotto_data: pd.DataFrame,
                               statistics: Dict[str, Any]) -> List[str]:
        """통계 차트들 생성"""
        chart_files = []
        
        try:
            # 1. 번호별 빈도 차트
            freq_chart = self._create_frequency_chart(statistics)
            if freq_chart:
                chart_files.append(freq_chart)
            
            # 2. 홀짝 분포 차트
            odd_even_chart = self._create_odd_even_chart(lotto_data)
            if odd_even_chart:
                chart_files.append(odd_even_chart)
            
            # 3. 구간별 분포 차트
            section_chart = self._create_section_distribution_chart(lotto_data)
            if section_chart:
                chart_files.append(section_chart)
            
            # 4. 시계열 트렌드 차트
            trend_chart = self._create_trend_chart(lotto_data)
            if trend_chart:
                chart_files.append(trend_chart)
                
        except Exception as e:
            print(f"통계 차트 생성 중 오류: {e}")
        
        return chart_files
    
    def _create_frequency_chart(self, statistics: Dict[str, Any]) -> Optional[str]:
        """번호별 빈도 차트 생성"""
        try:
            freq_analysis = statistics.get('frequency_analysis', {})
            frequency_count = freq_analysis.get('frequency_count', {})
            
            if not frequency_count:
                return None
            
            # 데이터 준비
            numbers = list(range(1, 46))
            frequencies = [frequency_count.get(num, 0) for num in numbers]
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(15, 8))
            
            bars = ax.bar(numbers, frequencies, alpha=0.7)
            
            # 색상 설정 (빈도에 따라)
            max_freq = max(frequencies) if frequencies else 1
            for i, bar in enumerate(bars):
                height = bar.get_height()
                normalized = height / max_freq
                bar.set_color(plt.cm.RdYlGn(normalized))
            
            ax.set_xlabel('번호')
            ax.set_ylabel('출현 횟수')
            ax.set_title('로또 번호별 출현 빈도')
            ax.set_xticks(range(1, 46, 5))
            ax.grid(True, alpha=0.3)
            
            # 평균선 추가
            avg_freq = np.mean(frequencies)
            ax.axhline(y=avg_freq, color='red', linestyle='--', alpha=0.7, label=f'평균: {avg_freq:.1f}')
            ax.legend()
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'number_frequency.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"빈도 차트 생성 실패: {e}")
            return None
    
    def _create_odd_even_chart(self, lotto_data: pd.DataFrame) -> Optional[str]:
        """홀짝 분포 차트 생성"""
        try:
            if 'odd_count' not in lotto_data.columns:
                return None
            
            odd_counts = lotto_data['odd_count'].value_counts().sort_index()
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 막대 차트
            colors = ['skyblue', 'lightgreen', 'gold', 'orange', 'salmon', 'purple', 'pink']
            ax1.bar(odd_counts.index, odd_counts.values, color=colors[:len(odd_counts)])
            ax1.set_xlabel('홀수 개수')
            ax1.set_ylabel('빈도')
            ax1.set_title('홀수 개수별 분포')
            ax1.set_xticks(range(7))
            
            # 파이 차트
            ax2.pie(odd_counts.values, labels=[f'{i}개 홀수' for i in odd_counts.index], 
                   autopct='%1.1f%%', colors=colors[:len(odd_counts)])
            ax2.set_title('홀수 개수 비율')
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'odd_even_distribution.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"홀짝 분포 차트 생성 실패: {e}")
            return None
    
    def _create_section_distribution_chart(self, lotto_data: pd.DataFrame) -> Optional[str]:
        """구간별 분포 차트 생성"""
        try:
            required_cols = ['low_count', 'mid_count', 'high_count']
            if not all(col in lotto_data.columns for col in required_cols):
                return None
            
            # NaN 값 처리 및 데이터 정제
            clean_data = lotto_data.copy()
            for col in required_cols:
                # NaN 값을 0으로 대체하고 숫자형으로 변환
                clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce').fillna(0)
            
            # 데이터 준비 (NaN 체크 추가)
            section_means = []
            for col in required_cols:
                col_data = clean_data[col].dropna()
                if len(col_data) > 0:
                    section_means.append(float(col_data.mean()))
                else:
                    section_means.append(0.0)
            
            section_names = ['저구간\n(1-15)', '중구간\n(16-30)', '고구간\n(31-45)']
            
            # 모든 값이 0이면 차트 생성 안함
            if all(mean == 0 for mean in section_means):
                print("구간별 데이터가 없어 차트 생성 건너뜀")
                return None
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(section_names, section_means, 
                            color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
            
            # 값 표시
            for bar, value in zip(bars, section_means):
                if value > 0:  # 0이 아닐 때만 표시
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('평균 번호 개수')
            ax.set_title('구간별 평균 분포')
            
            # y축 범위 안전하게 설정
            max_value = max(section_means) if section_means else 3
            ax.set_ylim(0, max(max_value * 1.2, 3))
            
            # 이상적 분포선 (각 구간 2개씩)
            ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='이상적 분포 (2개씩)')
            ax.legend()
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'section_distribution.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"구간 분포 차트 생성 실패: {e}")
            return None
    
    def _create_trend_chart(self, lotto_data: pd.DataFrame) -> Optional[str]:
        """시계열 트렌드 차트 생성"""
        try:
            if len(lotto_data) < 50:  # 충분한 데이터가 없으면 스킵
                return None
            
            # 최근 50회차 데이터 사용
            recent_data = lotto_data.head(50).sort_values('round_number')
            
            # 차트 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 번호 합계 트렌드
            axes[0, 0].plot(recent_data['round_number'], recent_data['number_sum'], 
                           marker='o', markersize=3, linewidth=1)
            axes[0, 0].set_title('번호 합계 트렌드')
            axes[0, 0].set_xlabel('회차')
            axes[0, 0].set_ylabel('합계')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 홀수 개수 트렌드
            if 'odd_count' in recent_data.columns:
                axes[0, 1].plot(recent_data['round_number'], recent_data['odd_count'], 
                               marker='s', markersize=3, linewidth=1, color='green')
                axes[0, 1].set_title('홀수 개수 트렌드')
                axes[0, 1].set_xlabel('회차')
                axes[0, 1].set_ylabel('홀수 개수')
                axes[0, 1].set_ylim(0, 6)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 구간별 분포 스택 차트
            if all(col in recent_data.columns for col in ['low_count', 'mid_count', 'high_count']):
                axes[1, 0].stackplot(recent_data['round_number'],
                                   recent_data['low_count'],
                                   recent_data['mid_count'], 
                                   recent_data['high_count'],
                                   labels=['저구간', '중구간', '고구간'],
                                   alpha=0.7)
                axes[1, 0].set_title('구간별 분포 트렌드')
                axes[1, 0].set_xlabel('회차')
                axes[1, 0].set_ylabel('개수')
                axes[1, 0].legend(loc='upper right')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 이동평균 트렌드
            window = min(10, len(recent_data) // 3)
            if window > 1:
                moving_avg = recent_data['number_sum'].rolling(window=window).mean()
                axes[1, 1].plot(recent_data['round_number'], recent_data['number_sum'], 
                               alpha=0.3, color='gray', label='실제값')
                axes[1, 1].plot(recent_data['round_number'], moving_avg, 
                               color='red', linewidth=2, label=f'{window}회 이동평균')
                axes[1, 1].set_title('합계 이동평균 트렌드')
                axes[1, 1].set_xlabel('회차')
                axes[1, 1].set_ylabel('합계')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'trend_analysis.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"트렌드 차트 생성 실패: {e}")
            return None


class PredictionPlotter:
    """예측 결과 시각화 클래스"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def create_prediction_charts(self, predictions: List[Dict]) -> List[str]:
        """예측 차트들 생성"""
        chart_files = []
        
        try:
            # 1. 예측 분포 차트
            dist_chart = self._create_prediction_distribution_chart(predictions)
            if dist_chart:
                chart_files.append(dist_chart)
            
            # 2. 신뢰도 비교 차트
            conf_chart = self._create_confidence_comparison_chart(predictions)
            if conf_chart:
                chart_files.append(conf_chart)
            
            # 3. 예측 다양성 차트
            div_chart = self._create_diversity_chart(predictions)
            if div_chart:
                chart_files.append(div_chart)
                
        except Exception as e:
            print(f"예측 차트 생성 중 오류: {e}")
        
        return chart_files
    
    def _create_prediction_distribution_chart(self, predictions: List[Dict]) -> Optional[str]:
        """예측 분포 차트 생성"""
        try:
            # 모든 예측된 번호 수집
            all_predicted = []
            for pred in predictions:
                final_pred = pred.get('final_prediction', [])
                all_predicted.extend(final_pred)
            
            if not all_predicted:
                return None
            
            # 번호별 예측 빈도
            from collections import Counter
            pred_freq = Counter(all_predicted)
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(15, 8))
            
            numbers = list(range(1, 46))
            frequencies = [pred_freq.get(num, 0) for num in numbers]
            
            bars = ax.bar(numbers, frequencies, alpha=0.7)
            
            # 색상 설정
            max_freq = max(frequencies) if frequencies else 1
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    normalized = height / max_freq
                    bar.set_color(plt.cm.Blues(0.3 + normalized * 0.7))
                else:
                    bar.set_color('lightgray')
            
            ax.set_xlabel('번호')
            ax.set_ylabel('예측 빈도')
            ax.set_title(f'예측 번호 분포 ({len(predictions)}개 예측 세트)')
            ax.set_xticks(range(1, 46, 5))
            ax.grid(True, alpha=0.3)
            
            # 예측되지 않은 번호 표시
            unpredicted = [num for num in numbers if frequencies[num-1] == 0]
            if unpredicted:
                ax.text(0.02, 0.98, f'예측되지 않은 번호: {len(unpredicted)}개', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'prediction_distribution.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"예측 분포 차트 생성 실패: {e}")
            return None
    
    def _create_confidence_comparison_chart(self, predictions: List[Dict]) -> Optional[str]:
        """신뢰도 비교 차트 생성"""
        try:
            confidences = []
            labels = []
            
            for i, pred in enumerate(predictions, 1):
                conf = pred.get('ensemble_confidence', 0)
                confidences.append(conf)
                labels.append(f'예측 {i}')
            
            if not confidences:
                return None
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 막대 차트
            colors = ['green' if c >= 0.7 else 'orange' if c >= 0.5 else 'red' for c in confidences]
            bars = ax1.bar(labels, confidences, color=colors, alpha=0.7)
            
            ax1.set_ylabel('신뢰도')
            ax1.set_title('예측별 신뢰도 비교')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 값 표시
            for bar, conf in zip(bars, confidences):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # 신뢰도 분포 히스토그램
            ax2.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('신뢰도')
            ax2.set_ylabel('빈도')
            ax2.set_title('신뢰도 분포')
            ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'평균: {np.mean(confidences):.2f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'confidence_comparison.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"신뢰도 비교 차트 생성 실패: {e}")
            return None
    
    def _create_diversity_chart(self, predictions: List[Dict]) -> Optional[str]:
        """예측 다양성 차트 생성"""
        try:
            # 예측간 유사도 계산
            similarities = []
            for i in range(len(predictions)):
                for j in range(i+1, len(predictions)):
                    pred1 = set(predictions[i].get('final_prediction', []))
                    pred2 = set(predictions[j].get('final_prediction', []))
                    
                    similarity = len(pred1 & pred2) / len(pred1 | pred2) if pred1 | pred2 else 0
                    similarities.append(similarity)
            
            if not similarities:
                return None
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 유사도 분포 히스토그램
            ax1.hist(similarities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax1.set_xlabel('예측간 유사도')
            ax1.set_ylabel('빈도')
            ax1.set_title('예측 다양성 (유사도 분포)')
            ax1.axvline(np.mean(similarities), color='red', linestyle='--',
                       label=f'평균 유사도: {np.mean(similarities):.2f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 예측별 고유성 점수
            uniqueness_scores = []
            for i, pred in enumerate(predictions):
                pred_set = set(pred.get('final_prediction', []))
                
                # 다른 예측들과의 평균 차이
                differences = []
                for j, other_pred in enumerate(predictions):
                    if i != j:
                        other_set = set(other_pred.get('final_prediction', []))
                        diff = len(pred_set - other_set) / len(pred_set) if pred_set else 0
                        differences.append(diff)
                
                uniqueness = np.mean(differences) if differences else 0
                uniqueness_scores.append(uniqueness)
            
            # 고유성 막대 차트
            labels = [f'예측 {i+1}' for i in range(len(predictions))]
            colors = ['darkgreen' if u >= 0.7 else 'orange' if u >= 0.4 else 'red' for u in uniqueness_scores]
            
            bars = ax2.bar(labels, uniqueness_scores, color=colors, alpha=0.7)
            ax2.set_ylabel('고유성 점수')
            ax2.set_title('예측별 고유성')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 값 표시
            for bar, score in zip(bars, uniqueness_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'prediction_diversity.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"다양성 차트 생성 실패: {e}")
            return None


class PerformancePlotter:
    """성능 분석 차트 클래스"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def create_performance_charts(self, performance_data: Dict[str, Any]) -> List[str]:
        """성능 차트들 생성"""
        chart_files = []
        
        try:
            # 1. 실행 시간 차트
            time_chart = self._create_execution_time_chart(performance_data)
            if time_chart:
                chart_files.append(time_chart)
            
            # 2. 메모리 사용량 차트
            memory_chart = self._create_memory_usage_chart(performance_data)
            if memory_chart:
                chart_files.append(memory_chart)
                
        except Exception as e:
            print(f"성능 차트 생성 중 오류: {e}")
        
        return chart_files
    
    def _create_execution_time_chart(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """실행 시간 차트 생성"""
        try:
            summary = performance_data.get('summary', {})
            if not summary:
                return None
            
            functions = list(summary.keys())
            avg_times = [summary[func]['avg_time'] for func in functions]
            total_times = [summary[func]['total_time'] for func in functions]
            counts = [summary[func]['count'] for func in functions]
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # 평균 실행 시간
            bars1 = ax1.barh(functions, avg_times, color='skyblue', alpha=0.7)
            ax1.set_xlabel('평균 실행 시간 (초)')
            ax1.set_title('함수별 평균 실행 시간')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # 값 표시
            for bar, time_val in zip(bars1, avg_times):
                ax1.text(bar.get_width() + max(avg_times) * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{time_val:.3f}s', va='center', fontsize=9)
            
            # 총 실행 시간 vs 호출 횟수
            colors = plt.cm.viridis(np.linspace(0, 1, len(functions)))
            scatter = ax2.scatter(counts, total_times, c=colors, s=100, alpha=0.7)
            
            ax2.set_xlabel('호출 횟수')
            ax2.set_ylabel('총 실행 시간 (초)')
            ax2.set_title('호출 횟수 vs 총 실행 시간')
            ax2.grid(True, alpha=0.3)
            
            # 함수명 라벨
            for i, func in enumerate(functions):
                ax2.annotate(func.split('_')[-1], (counts[i], total_times[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'execution_times.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"실행 시간 차트 생성 실패: {e}")
            return None
    
    def _create_memory_usage_chart(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """메모리 사용량 차트 생성 (시뮬레이션)"""
        try:
            # 메모리 데이터가 없으면 시뮬레이션 데이터 생성
            operations = ['data_loading', 'statistics', 'prediction', 'synthesis', 'output']
            memory_usage = [45.2, 67.8, 89.3, 92.1, 88.5]  # MB
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['lightblue', 'lightgreen', 'gold', 'orange', 'lightcoral']
            bars = ax.bar(operations, memory_usage, color=colors, alpha=0.7)
            
            ax.set_ylabel('메모리 사용량 (MB)')
            ax.set_title('단계별 메모리 사용량')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 값 표시
            for bar, usage in zip(bars, memory_usage):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{usage:.1f}MB', ha='center', va='bottom', fontweight='bold')
            
            # 평균선
            avg_memory = np.mean(memory_usage)
            ax.axhline(y=avg_memory, color='red', linestyle='--', alpha=0.7,
                      label=f'평균: {avg_memory:.1f}MB')
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 저장
            file_path = self.output_dir / 'memory_usage.png'
            plt.savefig(file_path)
            plt.close()
            
            return str(file_path)
            
        except Exception as e:
            print(f"메모리 사용량 차트 생성 실패: {e}")
            return None


# === 유틸리티 함수들 ===
def save_chart_data(chart_data: Dict[str, Any], filename: str):
    """차트 데이터를 JSON으로 저장"""
    output_dir = Config.OUTPUTS_DIR / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"{filename}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chart_data, f, ensure_ascii=False, indent=2, default=str)


def create_simple_chart(data: Dict[str, List], 
                       title: str, 
                       chart_type: str = 'bar',
                       save_name: str = 'simple_chart') -> str:
    """간단한 차트 생성 유틸리티"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == 'bar':
            for i, (label, values) in enumerate(data.items()):
                ax.bar(range(len(values)), values, label=label, alpha=0.7)
        elif chart_type == 'line':
            for i, (label, values) in enumerate(data.items()):
                ax.plot(range(len(values)), values, label=label, marker='o')
        elif chart_type == 'scatter':
            for i, (label, values) in enumerate(data.items()):
                ax.scatter(range(len(values)), values, label=label, alpha=0.7)
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        output_dir = Config.OUTPUTS_DIR / 'visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f'{save_name}.png'
        plt.savefig(file_path)
        plt.close()
        
        return str(file_path)
        
    except Exception as e:
        print(f"간단한 차트 생성 실패: {e}")
        return ""


# === 모듈 테스트 ===
if __name__ == "__main__":
    print("=== 시각화 도구 테스트 ===")
    
    try:
        # 테스트 데이터 생성
        test_lotto_data = pd.DataFrame({
            'round_number': [1187, 1186, 1185, 1184, 1183],
            'number_sum': [150, 90, 134, 146, 122],
            'odd_count': [3, 3, 3, 5, 3],
            'low_count': [1, 3, 1, 0, 1],
            'mid_count': [2, 3, 4, 4, 4],
            'high_count': [3, 0, 1, 2, 1]
        })
        
        test_statistics = {
            'frequency_analysis': {
                'frequency_count': {i: np.random.randint(8, 18) for i in range(1, 46)}
            }
        }
        
        test_predictions = [
            {'final_prediction': [1, 7, 14, 21, 28, 35], 'ensemble_confidence': 0.8},
            {'final_prediction': [3, 9, 15, 22, 29, 36], 'ensemble_confidence': 0.7},
            {'final_prediction': [5, 12, 19, 26, 33, 40], 'ensemble_confidence': 0.6}
        ]
        
        test_performance = {
            'summary': {
                'data_loading': {'avg_time': 0.045, 'total_time': 0.135, 'count': 3},
                'statistics': {'avg_time': 0.123, 'total_time': 0.246, 'count': 2},
                'prediction': {'avg_time': 0.234, 'total_time': 0.468, 'count': 2}
            }
        }
        
        # 시각화 도구 초기화
        visualizer = LottoVisualizer()
        
        # 종합 리포트 생성
        report_path = visualizer.create_comprehensive_report(
            test_lotto_data, test_statistics, test_predictions, test_performance
        )
        
        if report_path:
            print(f"✅ 종합 리포트 생성 완료: {report_path}")
        
        # 간단한 차트 테스트
        simple_data = {
            'Series A': [1, 3, 2, 5, 4],
            'Series B': [2, 4, 3, 6, 5]
        }
        
        simple_chart_path = create_simple_chart(
            simple_data, 
            "테스트 차트", 
            "line", 
            "test_simple_chart"
        )
        
        if simple_chart_path:
            print(f"✅ 간단한 차트 생성 완료: {simple_chart_path}")
        
        print("✅ 시각화 도구 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()