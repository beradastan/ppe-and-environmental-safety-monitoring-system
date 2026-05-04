-- ============================================================
-- PPE Güvenlik Sistemi — PostgreSQL Şeması
-- Idempotent: CREATE TABLE IF NOT EXISTS kullanılır
-- ============================================================

-- Her event'in güncel (en son) durumu
CREATE TABLE IF NOT EXISTS events (
    id               SERIAL       PRIMARY KEY,
    event_id         VARCHAR(20)  UNIQUE NOT NULL,   -- evt_0001
    event_status     VARCHAR(20)  NOT NULL,           -- new | active | update | closed
    created_at       TIMESTAMPTZ  NOT NULL,           -- ilk "new" zamanı
    updated_at       TIMESTAMPTZ  NOT NULL,           -- son güncelleme
    repeat_count     INT          NOT NULL DEFAULT 0,
    duration_sec     FLOAT        NOT NULL DEFAULT 0.0,
    -- İhlal bayrakları — hızlı filtreleme için denormalize
    helmet_violation BOOLEAN      NOT NULL DEFAULT FALSE,
    vest_violation   BOOLEAN      NOT NULL DEFAULT FALSE,
    mask_violation   BOOLEAN      NOT NULL DEFAULT FALSE,
    fire_detected    BOOLEAN      NOT NULL DEFAULT FALSE,
    -- Tam imza ve LLM raporu
    signature        JSONB,
    llm_report       TEXT,
    -- Kişi bazlı PPE detayı (track_id, durum, confidence)
    persons          JSONB,
    false_positive   BOOLEAN      NOT NULL DEFAULT FALSE
);

-- Her event'in tüm durum geçişleri (zaman çizgisi)
CREATE TABLE IF NOT EXISTS event_timeline (
    id             SERIAL       PRIMARY KEY,
    event_id       VARCHAR(20)  NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    event_status   VARCHAR(20)  NOT NULL,
    ts             TIMESTAMPTZ  NOT NULL,
    repeat_count   INT          NOT NULL DEFAULT 0,
    duration_sec   FLOAT        NOT NULL DEFAULT 0.0,
    change_reason  TEXT,
    signature      JSONB,
    llm_report     TEXT,
    image_filename VARCHAR(200),
    recorded_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    -- Kişi bazlı PPE detayı
    persons        JSONB
);

-- Operatör notları
CREATE TABLE IF NOT EXISTS event_notes (
    id          SERIAL       PRIMARY KEY,
    event_id    VARCHAR(20)  NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    note_text   TEXT         NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Mevcut tablolara kolon ekle (tablo zaten varsa CREATE TABLE IF NOT EXISTS atlanır)
ALTER TABLE events          ADD COLUMN IF NOT EXISTS persons        JSONB;
ALTER TABLE event_timeline  ADD COLUMN IF NOT EXISTS persons        JSONB;
ALTER TABLE events          ADD COLUMN IF NOT EXISTS camera_id      VARCHAR(20);
ALTER TABLE events          ADD COLUMN IF NOT EXISTS zone           VARCHAR(50);
ALTER TABLE events          ADD COLUMN IF NOT EXISTS false_positive BOOLEAN NOT NULL DEFAULT FALSE;

-- Otomatik ve manuel oluşturulan LLM raporları
CREATE TABLE IF NOT EXISTS llm_reports (
    id             SERIAL       PRIMARY KEY,
    period         VARCHAR(10)  NOT NULL,   -- daily | weekly | monthly
    report_date    VARCHAR(10)  NOT NULL,   -- YYYY-MM-DD (dönem başlangıcı)
    llm_text       TEXT         NOT NULL,
    generated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    auto_generated BOOLEAN      NOT NULL DEFAULT FALSE,
    UNIQUE (period, report_date)
);

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_events_status      ON events(event_status);
CREATE INDEX IF NOT EXISTS idx_events_updated_at  ON events(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_created_at  ON events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_helmet      ON events(helmet_violation);
CREATE INDEX IF NOT EXISTS idx_events_vest        ON events(vest_violation);
CREATE INDEX IF NOT EXISTS idx_events_mask        ON events(mask_violation);
CREATE INDEX IF NOT EXISTS idx_events_fire        ON events(fire_detected);
CREATE INDEX IF NOT EXISTS idx_timeline_event_id  ON event_timeline(event_id);
CREATE INDEX IF NOT EXISTS idx_timeline_ts        ON event_timeline(ts DESC);
CREATE INDEX IF NOT EXISTS idx_notes_event_id       ON event_notes(event_id);
CREATE INDEX IF NOT EXISTS idx_events_false_positive ON events(false_positive);
