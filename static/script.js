// 전역 설정
document.addEventListener('DOMContentLoaded', function() {
    // 스무스 스크롤
    initSmoothScroll();
    
    // 애니메이션 옵저버
    initAnimationObserver();
    
    // 툴팁 초기화
    initTooltips();
});

// 스무스 스크롤
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// 스크롤 애니메이션
function initAnimationObserver() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// 툴팁
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(el => {
        el.addEventListener('mouseenter', showTooltip);
        el.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(e) {
    const tooltipText = e.target.getAttribute('data-tooltip');
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = tooltipText;
    
    document.body.appendChild(tooltip);
    
    const rect = e.target.getBoundingClientRect();
    tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
    tooltip.style.left = rect.left + (rect.width - tooltip.offsetWidth) / 2 + 'px';
    
    setTimeout(() => tooltip.classList.add('show'), 10);
}

function hideTooltip(e) {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.classList.remove('show');
        setTimeout(() => tooltip.remove(), 300);
    }
}

// 차트 네비게이션
function showChart(chartId) {
    // 탭 버튼 활성화
    document.querySelectorAll('.chart-tabs .tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // 차트 표시
    document.querySelectorAll('.chart-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(chartId).classList.add('active');
}

// 통계 네비게이션
function showStats(statType) {
    // 탭 버튼 활성화
    document.querySelectorAll('.stats-tabs .tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // 통계 표시
    document.querySelectorAll('.stat-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(statType + '-stats').classList.add('active');
}

// 공유 기능
function sharePrediction() {
    const shareData = {
        title: '연금복권 AI 예측 결과',
        text: document.querySelector('.share-text').textContent,
        url: window.location.href
    };
    
    if (navigator.share) {
        navigator.share(shareData)
            .then(() => showNotification('공유되었습니다!', 'success'))
            .catch(() => showNotification('공유 취소', 'info'));
    } else {
        // 클립보드 복사
        const text = shareData.text + ' ' + shareData.url;
        navigator.clipboard.writeText(text)
            .then(() => showNotification('클립보드에 복사되었습니다!', 'success'));
    }
}

// 알림 표시
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 10);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// API 호출 헬퍼
async function fetchPrediction() {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) throw new Error('예측 실패');
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        showNotification('예측 중 오류가 발생했습니다.', 'error');
        return null;
    }
}

// 로딩 표시
function showLoading(show = true) {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.style.display = show ? 'block' : 'none';
    }
}

// 숫자 애니메이션
function animateNumber(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// 모바일 메뉴 토글
function toggleMobileMenu() {
    const navMenu = document.querySelector('.nav-menu');
    navMenu.classList.toggle('active');
}

// 키보드 단축키
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + P: 예측하기
    if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        e.preventDefault();
        window.location.href = '/predict';
    }
    
    // Ctrl/Cmd + H: 홈
    if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
        e.preventDefault();
        window.location.href = '/';
    }
});

// 개발자 콘솔 메시지
console.log('%c🎰 연금복권 AI 예측 시스템', 'font-size: 24px; font-weight: bold; color: #667eea;');
console.log('%c이 시스템은 교육 및 연구 목적으로 제작되었습니다.', 'font-size: 14px; color: #666;');
console.log('%c실제 당첨을 보장하지 않습니다.', 'font-size: 14px; color: #f56565;');