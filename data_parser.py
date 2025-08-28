from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import requests
import json

class LottoDataParser:
    def __init__(self, html_path=None):
        self.html_path = html_path
        self.data = None
    
    def parse_html(self):
        """HTML 파일에서 데이터 추출"""
        with open(self.html_path, 'r', encoding='euc-kr') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        data = []
        table = soup.find_all('table')[1]
        rows = table.find_all('tr')[2:]
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 11:
                # rowspan 처리를 위한 year 관리
                if cols[0].get('rowspan'):
                    current_year = cols[0].text.strip()
                    col_offset = 0
                else:
                    col_offset = -1
                
                draw_no = int(cols[1+col_offset].text.strip())
                draw_date = cols[2+col_offset].text.strip()
                
                # 1등 번호 추출
                first_prize = cols[3+col_offset].text.strip()
                match = re.search(r'(\d)조\s*(\d+)', first_prize)
                if match:
                    jo = int(match.group(1))
                    number = match.group(2).zfill(6)
                else:
                    continue
                
                # 보너스 번호
                bonus = cols[10+col_offset].text.strip()
                
                data.append({
                    'draw_no': draw_no,
                    'draw_date': draw_date,
                    'jo': jo,
                    'number': number,
                    'bonus_number': bonus
                })
        
        self.data = pd.DataFrame(data)
        self.data['draw_date'] = pd.to_datetime(self.data['draw_date'], format='%Y%m%d')
        self.data = self.data.sort_values('draw_no')
        return self.data
    
    def fetch_latest_data(self):
        """최신 데이터 크롤링 (실제 구현 시 해당 사이트 API 사용)"""
        # 이것은 예시 코드입니다. 실제로는 공식 API를 사용해야 합니다.
        try:
            # API URL은 실제 연금복권 API로 교체 필요
            url = "https://api.example.com/pension/latest"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None