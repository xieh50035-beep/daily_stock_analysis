import logging
from datetime import datetime, timedelta
from math import prod

import pandas as pd

from data_provider.akshare_fetcher import fetch_sw_industry_daily
from src.core.trading_calendar import (
    get_effective_trading_date,
    get_market_now,
    get_trading_dates,
)


logger = logging.getLogger(__name__)
BENCHMARKS = (
    ("sh000001", "上证指数"),
    ("sz399001", "深证成指"),
    ("sz399006", "创业板指"),
)
SW_INDUSTRY_COUNT = 31


def _session_values(frame, column, sessions, *, positive=False):
    """Require one valid observation for each expected session."""
    if not len(sessions) or column not in frame:
        return None
    values = frame[column].reindex(sessions)
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().any() or values.isin([float("inf"), -float("inf")]).any():
        return None
    if positive and (values <= 0).any():
        return None
    return values


def _index_history(fetcher, code, start, end):
    try:
        frame, source = fetcher.get_daily_data(
            code, start_date=start.isoformat(), end_date=end.isoformat(),
        )
        frame = frame.copy()
        frame.index = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dt.normalize()
        in_window = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        return frame.loc[in_window & ~frame.index.duplicated(keep=False)], source
    except Exception as exc:
        logger.warning("Weekly index history unavailable for %s (%s)", code, type(exc).__name__)
        return pd.DataFrame(), ""


def _industry_rankings(frame, sessions):
    columns = {"指数代码", "指数名称", "发布日期", "涨跌幅"}
    if not columns.issubset(frame.columns) or not len(sessions):
        return []
    frame = frame.assign(session=pd.to_datetime(frame["发布日期"], errors="coerce", format="mixed").dt.normalize())
    frame = frame.loc[frame["session"].isin(sessions)]
    rankings = []
    for _, group in frame.groupby("指数代码"):
        if not group["session"].is_unique:
            continue
        changes = _session_values(group.set_index("session"), "涨跌幅", sessions)
        if changes is None or (changes <= -100).any():
            continue
        name = group.loc[group["session"] == sessions[-1], "指数名称"].iloc[0]
        if pd.isna(name) or not str(name).strip():
            continue
        cumulative = (prod(1 + changes / 100) - 1) * 100
        rankings.append((str(name), cumulative, int((changes > 0).sum()), changes.iloc[-1]))
    return sorted(rankings, key=lambda row: row[1], reverse=True)


def build_weekly_market_report(period, reference_time: datetime, fetcher, stock_codes, results):
    """Build the scheduled A-share recap from dated observations, with explicit coverage."""
    midweek = period == "midweek"
    title = "周中趋势观察报告" if midweek else "本周复盘与下周展望"
    watch_window = "周四、周五" if midweek else "下周"
    local_date = get_market_now("cn", current_time=reference_time).date()
    weekday = 2 if midweek else 4
    cutoff = local_date - timedelta(days=(local_date.weekday() - weekday) % 7)
    monday = cutoff - timedelta(days=weekday)
    prior_monday = monday - timedelta(days=7)
    prior_friday = monday - timedelta(days=3)
    sessions = get_trading_dates("cn", monday, cutoff)
    prior = get_trading_dates("cn", prior_monday, prior_friday)
    closed_through = min(cutoff, get_effective_trading_date("cn", current_time=reference_time))

    warnings = []
    changes = []
    since_midweek = []
    rows = []
    conditions = []
    rankings = []
    if sessions is None or not len(sessions):
        warnings.append("本期交易日历缺失或无交易日，无法验证行情覆盖。")
    else:
        if prior is None or not len(prior):
            warnings.append("上周交易日历缺失或无交易日，无法核对涨跌基准和量能对照。")
        for code, name in BENCHMARKS:
            frame, source = _index_history(fetcher, code, prior_monday, closed_through)
            closes = _session_values(frame, "close", sessions, positive=True)
            baseline = (
                _session_values(frame, "close", prior[-1:], positive=True)
                if prior is not None else None
            )
            volume = _session_values(frame, "volume", sessions, positive=True)
            prior_volume = (
                _session_values(frame, "volume", prior, positive=True)
                if prior is not None else None
            )
            change = average = total = "缺失"
            if closes is not None and baseline is not None:
                change = f"{(closes.iloc[-1] / baseline.iloc[-1] - 1):+.2%}"
                changes.append(f"{name} {change}")
            if not midweek and closes is not None:
                first_half = closes.loc[:pd.Timestamp(monday + timedelta(days=2))]
                if len(first_half) and first_half.index[-1] < closes.index[-1]:
                    since_midweek.append(f"{name} {(closes.iloc[-1] / first_half.iloc[-1] - 1):+.2%}")
            if volume is not None and prior_volume is not None:
                average = f"{(volume.mean() / prior_volume.mean() - 1):+.2%}"
                total = f"{(volume.sum() / prior_volume.sum() - 1):+.2%}"
            if "缺失" in (change, average):
                warnings.append(f"{name}缺少完整的收盘基准、交易日收盘价或成交量，缺项见表。")
            total_cell = "" if midweek else f" {total} |"
            rows.append(f"| {name} | {change} | {average} |{total_cell} {source or '缺失'} |")
            if code == BENCHMARKS[0][0] and closes is not None:
                high, low = closes.max(), closes.min()
                conditions.append(
                    f"- 上证指数收盘站上本期最高收盘 {high:.2f} 点时观察强势能否延续；"
                    f"收盘跌破本期最低收盘 {low:.2f} 点时留意转弱。"
                )
                if prior_volume is not None:
                    conditions.append("- 用上周日均成交量作量能参照。突破时若成交量仍低于该参照，继续观察确认。")

        if sessions[-1].date() <= closed_through:
            try:
                frame = fetch_sw_industry_daily(sessions[0].date(), sessions[-1].date())
                rankings = _industry_rankings(frame, sessions)
            except Exception as exc:
                logger.warning("Weekly SW industry history unavailable (%s)", type(exc).__name__)
        if len(rankings) != SW_INDUSTRY_COUNT:
            warnings.append(
                f"申万一级行业完整日线覆盖 {len(rankings)}/{SW_INDUSTRY_COUNT}，"
                "行业排名仅覆盖数据完整的行业。"
            )

    failed_codes = sorted(set(stock_codes or ()) - {result.code for result in results})
    if failed_codes:
        warnings.append(f"个股分析未完成（行情或 AI 失败，详见运行日志）：{'、'.join(failed_codes)}。")
    no_news = [result.code for result in results if not result.news_evidence_present]
    if no_news:
        warnings.append(f"以下个股未取得可用消息面证据：{'、'.join(no_news)}。")
    no_close = [
        result.code for result in results
        if result.market_snapshot and result.market_snapshot.get("close") in (None, "N/A", "nan")
    ]
    if no_close:
        warnings.append(f"以下个股行情快照缺少收盘价：{'、'.join(no_close)}。")

    critical = bool(failed_codes or not changes)
    leaders = sorted(results, key=lambda result: result.sentiment_score, reverse=True)[:3]
    watchlist = "、".join(f"{result.name}（{result.code}）{result.sentiment_score} 分" for result in leaders)
    market_summary = "，".join(changes) or "指数区间涨跌缺失"
    sector_summary = ""
    if rankings:
        name, change, up_days, _ = rankings[0]
        sector_summary = f"。已覆盖行业中，{name}居首（{change:+.2f}%，{up_days}/{len(sessions)} 日上涨）"
    late_week_summary = f"。周中后变化为{'，'.join(since_midweek)}" if since_midweek else ""
    status = "🔴 关键数据或分析缺失" if critical else "🟡 部分数据缺失" if warnings else "🟢 已生成"
    report = [
        f"# {title} · {cutoff.isoformat()}",
        "## 执行摘要",
        f"- **状态** {status}。行情窗口为 {monday.isoformat()} 至 {cutoff.isoformat()}，"
        f"已收盘数据最晚截至 {closed_through.isoformat()}。",
        f"- **变化** {market_summary}{late_week_summary}{sector_summary}。",
        f"- **重点个股** {watchlist or ('暂无成功分析的个股评分' if stock_codes else '本次无个股目标')}。",
        f"- **风险** {' '.join(warnings) if warnings else '观察条件须由后续行情确认。'}",
        "",
        "## 指数涨跌与量能",
        "累计涨跌以上周最后一个交易日收盘为基准。量能为各指数成交量与上周比较，日均口径用于比较不同交易日数。",
    ]
    if rows:
        total_heading = "" if midweek else " 总量较上周 |"
        total_alignment = "" if midweek else " ---: |"
        report += [
            f"| 指数 | 本期累计涨跌 | 日均量较上周 |{total_heading} 日线来源 |",
            f"| --- | ---: | ---: |{total_alignment} --- |",
            *rows,
        ]
    else:
        report.append("缺失。")
    report += ["", "## 行业表现与持续性"]
    if rankings:
        report += [
            "来源为申万一级行业日报表（AKShare）。区间涨跌由各交易日日涨跌复合计算。",
            "| 期内居前行业 | 累计涨跌 | 上涨日 / 交易日 | 最后交易日 |",
            "| --- | ---: | ---: | ---: |",
            *(f"| {name} | {change:+.2f}% | {up}/{len(sessions)} | {last:+.2f}% |"
              for name, change, up, last in rankings[:5]),
        ]
        conditions.append("- 观察期内领先行业是否继续收涨并维持相对排名。若涨幅集中于少数交易日，降低对持续性的判断。")
    else:
        report.append("缺失，无法判断行业持续性。")
    report += ["", f"## {watch_window}观察条件", *(conditions or ["- 等待有效行情补齐后再判断。"])]
    return "\n".join(report) + "\n", critical, warnings
