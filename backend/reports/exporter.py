# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import io
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def generate_csv(events: list[dict], period: str, start: str, end: str) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        'Olay ID', 'Baslangic Tarihi', 'Bitis Tarihi',
        'Kamera', 'Bolge', 'Durum',
        'Baret Ihlali', 'Yelek Ihlali', 'Maske Ihlali', 'Yangin',
        'Sure (sn)', 'Tekrar Sayisi',
    ])

    for e in events:
        def _ts(val):
            if not val:
                return ''
            return str(val).replace('T', ' ')[:19]

        viols = []
        if e.get('helmet_violation'): viols.append('Baret')
        if e.get('vest_violation'):   viols.append('Yelek')
        if e.get('mask_violation'):   viols.append('Maske')
        if e.get('fire_detected'):    viols.append('Yangin')

        writer.writerow([
            e['event_id'],
            _ts(e.get('created_at')),
            _ts(e.get('updated_at')),
            e.get('camera_id') or '',
            e.get('zone') or '',
            e['event_status'],
            'Evet' if e.get('helmet_violation') else 'Hayir',
            'Evet' if e.get('vest_violation')   else 'Hayir',
            'Evet' if e.get('mask_violation')   else 'Hayir',
            'Evet' if e.get('fire_detected')    else 'Hayir',
            f"{float(e.get('duration_sec', 0)):.1f}",
            e.get('repeat_count', 0),
        ])

    return ('﻿' + output.getvalue()).encode('utf-8')


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _register_font():
    """Arial (Windows) kaydeder; yoksa Helvetica döner."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    win_fonts = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
    regular = os.path.join(win_fonts, 'arial.ttf')
    bold    = os.path.join(win_fonts, 'arialbd.ttf')

    if os.path.exists(regular) and os.path.exists(bold):
        try:
            pdfmetrics.registerFont(TTFont('AppFont',      regular))
            pdfmetrics.registerFont(TTFont('AppFont-Bold', bold))
            return 'AppFont', 'AppFont-Bold'
        except Exception:
            pass
    return 'Helvetica', 'Helvetica-Bold'


def generate_pdf(events: list[dict], period: str, start: str, end: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    FONT, FONT_BOLD = _register_font()

    PERIOD_LABELS = {'daily': 'Gunluk', 'weekly': 'Haftalik', 'monthly': 'Aylik'}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=2*cm,     bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    def _style(name, **kw):
        return ParagraphStyle(name, parent=styles['Normal'], fontName=FONT, **kw)

    title_s    = _style('T', fontName=FONT_BOLD, fontSize=18,
                        textColor=colors.HexColor('#1e293b'), spaceAfter=3)
    subtitle_s = _style('ST', fontSize=10,
                        textColor=colors.HexColor('#64748b'), spaceAfter=2)
    meta_s     = _style('MT', fontSize=8,
                        textColor=colors.HexColor('#94a3b8'), spaceAfter=10)
    section_s  = _style('SEC', fontName=FONT_BOLD, fontSize=11,
                        textColor=colors.HexColor('#334155'),
                        spaceBefore=14, spaceAfter=6)
    note_s     = _style('NOTE', fontSize=7.5,
                        textColor=colors.HexColor('#94a3b8'), spaceBefore=4)

    story = []

    # ── Başlık ──────────────────────────────────────────────
    story.append(Paragraph('SafetyMonitor', title_s))
    story.append(Paragraph(
        f'{PERIOD_LABELS.get(period, period)} Guvenlik Raporu  |  {start}  -  {end}',
        subtitle_s,
    ))
    story.append(Paragraph(
        f'Olusturulma: {datetime.now().strftime("%d.%m.%Y  %H:%M")}    '
        f'Toplam kayit: {len(events)}',
        meta_s,
    ))
    story.append(HRFlowable(
        width='100%', thickness=1.5,
        color=colors.HexColor('#3b82f6'), spaceAfter=12,
    ))

    # ── Özet istatistik tablosu ──────────────────────────────
    total    = len(events)
    closed   = sum(1 for e in events if e['event_status'] == 'closed')
    helmets  = sum(1 for e in events if e.get('helmet_violation'))
    vests    = sum(1 for e in events if e.get('vest_violation'))
    masks    = sum(1 for e in events if e.get('mask_violation'))
    fires    = sum(1 for e in events if e.get('fire_detected'))
    close_r  = f'{closed/total*100:.0f}%' if total else '-'

    story.append(Paragraph('Ozet', section_s))

    hdr_bg  = colors.HexColor('#1e40af')
    hdr_fg  = colors.white
    val_bg  = colors.HexColor('#eff6ff')
    border  = colors.HexColor('#bfdbfe')

    stat_data = [
        ['Donem Olayi', 'Kapanan', 'Baret', 'Yelek', 'Maske', 'Yangin'],
        [str(total), close_r, str(helmets), str(vests), str(masks), str(fires)],
    ]
    stat_table = Table(stat_data, colWidths=[2.9*cm]*6)
    stat_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), hdr_bg),
        ('TEXTCOLOR',     (0,0), (-1,0), hdr_fg),
        ('FONTNAME',      (0,0), (-1,0), FONT_BOLD),
        ('FONTSIZE',      (0,0), (-1,0), 8),
        ('BACKGROUND',    (0,1), (-1,1), val_bg),
        ('FONTNAME',      (0,1), (-1,1), FONT_BOLD),
        ('FONTSIZE',      (0,1), (-1,1), 14),
        ('TEXTCOLOR',     (0,1), (-1,1), colors.HexColor('#1e293b')),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('BOX',           (0,0), (-1,-1), 0.5, border),
        ('INNERGRID',     (0,0), (-1,-1), 0.5, border),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 4))

    # Ihlal yüzde çubuğu (metin tabanlı)
    if total:
        bar_data = [['Ihlal Tipi', 'Sayi', 'Oran', 'Dagilim']]
        for label, count in [('Baret', helmets), ('Yelek', vests),
                              ('Maske', masks),  ('Yangin', fires)]:
            pct = count / total
            filled = int(pct * 20)
            bar = '█' * filled + '░' * (20 - filled)
            bar_data.append([label, str(count), f'{pct*100:.1f}%', bar])

        bar_table = Table(bar_data, colWidths=[2.5*cm, 1.5*cm, 1.8*cm, 11.8*cm])
        bar_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#334155')),
            ('TEXTCOLOR',     (0,0), (-1,0), hdr_fg),
            ('FONTNAME',      (0,0), (-1,0), FONT_BOLD),
            ('FONTNAME',      (0,1), (-1,-1), FONT),
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('ALIGN',         (1,0), (2,-1), 'CENTER'),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 5),
            ('ROWBACKGROUNDS',(0,1), (-1,-1),
             [colors.white, colors.HexColor('#f8fafc')]),
            ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
        ]))
        story.append(bar_table)

    # ── Olay tablosu ────────────────────────────────────────
    story.append(Paragraph('Olay Listesi', section_s))

    ev_headers = ['Olay ID', 'Tarih', 'Kamera', 'Bolge',
                  'Durum', 'Ihlaller', 'Sure', 'Tekrar']
    col_w = [2.4*cm, 3.0*cm, 1.8*cm, 3.2*cm,
             1.8*cm, 3.2*cm, 1.4*cm, 1.4*cm]

    STATUS_TR = {'new': 'Yeni', 'active': 'Aktif', 'closed': 'Kapandi'}

    ev_data = [ev_headers]
    display = events[:300]
    for e in display:
        viols = []
        if e.get('helmet_violation'): viols.append('Baret')
        if e.get('vest_violation'):   viols.append('Yelek')
        if e.get('mask_violation'):   viols.append('Maske')
        if e.get('fire_detected'):    viols.append('Yangin')

        ts = str(e.get('created_at', ''))
        ts = ts.replace('T', ' ')[:16]

        ev_data.append([
            e['event_id'],
            ts,
            e.get('camera_id') or '-',
            (e.get('zone') or '-')[:20],
            STATUS_TR.get(e['event_status'], e['event_status']),
            ', '.join(viols) if viols else '-',
            f"{float(e.get('duration_sec', 0)):.0f}s",
            str(e.get('repeat_count', 0)),
        ])

    row_bgs = []
    for i in range(1, len(ev_data)):
        bg = colors.white if i % 2 else colors.HexColor('#f8fafc')
        row_bgs.append(('BACKGROUND', (0, i), (-1, i), bg))

    ev_table = Table(ev_data, colWidths=col_w, repeatRows=1)
    ev_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#334155')),
        ('TEXTCOLOR',     (0,0), (-1,0), hdr_fg),
        ('FONTNAME',      (0,0), (-1,0), FONT_BOLD),
        ('FONTSIZE',      (0,0), (-1,0), 8),
        ('FONTNAME',      (0,1), (-1,-1), FONT),
        ('FONTSIZE',      (0,1), (-1,-1), 7.5),
        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
    ] + row_bgs))
    story.append(ev_table)

    if len(events) > 300:
        story.append(Paragraph(
            f'* Tabloda ilk 300 kayit gosterilmektedir. Toplam: {len(events)}',
            note_s,
        ))

    doc.build(story)
    return buf.getvalue()
