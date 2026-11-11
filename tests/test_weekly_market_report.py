from argparse import Namespace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import main
from src.notification import ChannelAttemptResult, NotificationDispatchResult
from src.weekly_market_report import build_weekly_market_report


def reference(day, hour=10):
    return datetime(2025, 7, day, hour, tzinfo=timezone.utc)


@pytest.fixture
def history(monkeypatch):
    dates = pd.bdate_range("2025-06-30", "2025-07-11")
    frame = pd.DataFrame({
        "date": dates,
        "close": [100, 100, 100, 100, 100, 101, 102, 103, 104, 106],
        "volume": [100] * 5 + [120] * 5,
    })
    fetcher = Mock()
    fetcher.get_daily_data.return_value = (frame, "fixture")
    industry = pd.DataFrame([
        {"指数代码": code, "指数名称": f"行业{code}", "发布日期": day, "涨跌幅": change}
        for code in range(31)
        for day, change in zip(dates[5:], [1, -1, 2, 1, 2] if code == 0 else [0] * 5)
    ])
    monkeypatch.setattr("src.weekly_market_report.fetch_sw_industry_daily", lambda *_: industry)
    return fetcher, frame, industry


@pytest.mark.parametrize(
    "period, day, title, conditions, index_change, industry_change, up_days",
    [
        ("midweek", 9, "周中趋势观察报告", "周四、周五观察条件", "+3.00%", "+1.99%", "2/3"),
        ("weekly", 11, "本周复盘与下周展望", "下周观察条件", "+6.00%", "+5.07%", "4/5"),
    ],
)
def test_market_report_uses_expected_closed_sessions(
    history, period, day, title, conditions, index_change, industry_change, up_days,
):
    fetcher, _, _ = history
    report, critical, warnings = build_weekly_market_report(period, reference(day), fetcher, [], [])

    assert not critical and not warnings
    assert report.startswith(f"# {title} · 2025-07-{day:02}\n## 执行摘要")
    assert f"| 上证指数 | {index_change} | +20.00% |" in report
    assert f"| 行业0 | {industry_change} | {up_days} |" in report
    assert f"## {conditions}" in report
    assert ("总量较上周" in report) == (period == "weekly")
    assert ("周中后变化" in report) == (period == "weekly")
    assert fetcher.get_daily_data.call_args.kwargs["end_date"] == f"2025-07-{day:02}"


def test_unclosed_or_missing_sessions_cannot_be_substituted(history):
    fetcher, frame, industry = history
    # The provider returns a Friday bar despite the request ending on Thursday.
    report, critical, _ = build_weekly_market_report("weekly", reference(11, 4), fetcher, [], [])
    assert critical
    assert "| 上证指数 | 缺失 |" in report
    assert "缺失，无法判断行业持续性" in report

    # A missing interior session invalidates even a complete-looking endpoint pair.
    frame.drop(index=6, inplace=True)
    industry.drop(index=1, inplace=True)
    report, critical, warnings = build_weekly_market_report("weekly", reference(11), fetcher, [], [])
    assert critical
    assert "| 上证指数 | 缺失 |" in report
    assert "行业0 |" not in report
    assert any("30/31" in warning for warning in warnings)


def test_missing_calendar_is_critical_without_fetching(history, monkeypatch):
    fetcher, _, _ = history
    monkeypatch.setattr("src.weekly_market_report.get_trading_dates", lambda *_: None)
    report, critical, _ = build_weekly_market_report("weekly", reference(11), fetcher, [], [])

    assert critical
    assert "交易日历缺失" in report
    fetcher.get_daily_data.assert_not_called()


def test_scheduled_path_delivers_degraded_report_before_failing(history, monkeypatch):
    fetcher, _, _ = history
    config = SimpleNamespace(
        market_review_enabled=True, single_stock_notify=False, merge_email_notification=False,
        daily_market_context_enabled=False,
    )
    args = Namespace(no_market_review=False, dry_run=False, no_notify=False, workers=1, report_period="weekly")
    pipeline = Mock(fetcher_manager=fetcher)
    pipeline.run.return_value = []
    pipeline.notifier.send_with_results.return_value = NotificationDispatchResult(True, True, "sent")
    monkeypatch.setattr("main._refresh_stock_index_cache_for_analysis", Mock())
    monkeypatch.setattr("main._compute_trading_day_filter", lambda *_: (["600519"], "cn", False))
    monkeypatch.setattr("main._run_auto_backtest", Mock())
    monkeypatch.setattr("main.datetime", Mock(now=Mock(return_value=reference(11))))
    monkeypatch.setattr("src.core.pipeline.StockAnalysisPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr("src.feishu_doc.FeishuDocManager", Mock(return_value=Mock(is_configured=lambda: False)))

    assert main.run_full_analysis(config, args, ["600519"]) is False

    saved_report, filename = pipeline.notifier.save_report_to_file.call_args.args
    assert filename == "market_weekly_20250711.md"
    assert "个股分析未完成（行情或 AI 失败，详见运行日志）：600519" in saved_report
    assert "| 上证指数 | +6.00% |" in saved_report
    pipeline.notifier.send_with_results.assert_called_once_with(
        saved_report, email_send_to_all=True, route_type="report",
    )
    assert main._LAST_ANALYSIS_FAILURE_REASON == "weekly_report_incomplete"

    # The existing trading-day gate runs before collecting or sending the report.
    pipeline.reset_mock()
    monkeypatch.setattr("main._compute_trading_day_filter", lambda *_: ([], "", True))
    assert main.run_full_analysis(config, args, ["600519"]) is True
    pipeline.run.assert_not_called()
    pipeline.notifier.send_with_results.assert_not_called()


def test_successful_secondary_channel_does_not_mask_email_failure():
    notifier = Mock()
    notifier.send_with_results.return_value = NotificationDispatchResult(
        True, True, "partial_failed",
        [ChannelAttemptResult("email", False), ChannelAttemptResult("telegram", True)],
    )
    assert main._send_weekly_report(notifier, "report") is False
