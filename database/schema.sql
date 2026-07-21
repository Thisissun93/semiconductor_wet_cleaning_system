PRAGMA foreign_keys = ON;

-- =========================================================
-- 1. 설비 기준정보
-- =========================================================
CREATE TABLE IF NOT EXISTS equipment_master (
    equipment_id TEXT PRIMARY KEY,
    equipment_name TEXT NOT NULL,
    equipment_type TEXT NOT NULL DEFAULT 'WET_CLEANER',
    location TEXT,
    install_date TEXT,
    status TEXT NOT NULL DEFAULT 'RUN'
        CHECK (status IN ('RUN', 'IDLE', 'PM', 'DOWN'))
);

-- =========================================================
-- 2. Recipe 기준정보
-- =========================================================
CREATE TABLE IF NOT EXISTS recipe_master (
    recipe_id TEXT PRIMARY KEY,
    recipe_name TEXT NOT NULL,
    chemical_type TEXT NOT NULL,
    target_concentration REAL NOT NULL,
    target_bath_temperature REAL NOT NULL,
    target_cleaning_time REAL NOT NULL,
    target_rinse_time REAL NOT NULL,
    target_drying_time REAL NOT NULL,
    target_megasonic_power REAL NOT NULL,
    target_spin_speed REAL NOT NULL,
    target_diw_resistivity REAL NOT NULL
);

-- =========================================================
-- 3. 약액 LOT 정보
-- =========================================================
CREATE TABLE IF NOT EXISTS chemical_lot (
    chemical_lot_id TEXT PRIMARY KEY,
    chemical_type TEXT NOT NULL,
    supplier TEXT,
    manufacture_date TEXT,
    expiry_date TEXT,
    purity REAL NOT NULL,
    metal_contamination_ppb REAL NOT NULL,
    particle_level REAL NOT NULL,
    ph REAL NOT NULL,
    conductivity REAL NOT NULL,
    coa_result TEXT NOT NULL
        CHECK (coa_result IN ('PASS', 'FAIL'))
);

-- =========================================================
-- 4. 생산 LOT 정보
-- =========================================================
CREATE TABLE IF NOT EXISTS lot_history (
    lot_id TEXT PRIMARY KEY,
    product_type TEXT NOT NULL,
    process_step TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    chemical_lot_id TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    lot_status TEXT NOT NULL
        CHECK (lot_status IN ('PASS', 'FAIL', 'HOLD')),

    FOREIGN KEY (recipe_id)
        REFERENCES recipe_master(recipe_id),

    FOREIGN KEY (equipment_id)
        REFERENCES equipment_master(equipment_id),

    FOREIGN KEY (chemical_lot_id)
        REFERENCES chemical_lot(chemical_lot_id)
);

-- =========================================================
-- 5. Wafer 정보
-- 한 LOT에 여러 Wafer가 존재하므로 별도 테이블로 분리
-- =========================================================
CREATE TABLE IF NOT EXISTS wafer_history (
    wafer_id TEXT PRIMARY KEY,
    lot_id TEXT NOT NULL,
    wafer_number INTEGER NOT NULL,
    wafer_status TEXT NOT NULL
        CHECK (wafer_status IN ('PASS', 'FAIL', 'HOLD')),

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE
);

-- =========================================================
-- 6. LOT별 실제 공정조건
-- Recipe 목표값과 실제 실행값을 비교하기 위한 테이블
-- =========================================================
CREATE TABLE IF NOT EXISTS process_condition (
    process_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL UNIQUE,
    chemical_concentration REAL NOT NULL,
    bath_temperature REAL NOT NULL,
    cleaning_time REAL NOT NULL,
    rinse_time REAL NOT NULL,
    drying_time REAL NOT NULL,
    megasonic_power REAL NOT NULL,
    spin_speed REAL NOT NULL,
    diw_resistivity REAL NOT NULL,
    bath_age_hours REAL NOT NULL,

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE
);

-- =========================================================
-- 7. 설비 센서 데이터
-- 한 LOT 처리 중 여러 시점의 센서 데이터 저장 가능
-- =========================================================
CREATE TABLE IF NOT EXISTS sensor_history (
    sensor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    measured_at TEXT NOT NULL,
    pump_pressure REAL NOT NULL,
    flow_rate REAL NOT NULL,
    filter_differential_pressure REAL NOT NULL,
    nozzle_pressure REAL NOT NULL,
    motor_current REAL NOT NULL,
    vibration REAL NOT NULL,
    exhaust_pressure REAL NOT NULL,
    alarm_code TEXT,

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE,

    FOREIGN KEY (equipment_id)
        REFERENCES equipment_master(equipment_id)
);

-- =========================================================
-- 8. 품질 결과
-- =========================================================
CREATE TABLE IF NOT EXISTS quality_result (
    quality_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL UNIQUE,
    particle_count INTEGER NOT NULL,
    metal_contamination_ppb REAL NOT NULL,
    water_mark_count INTEGER NOT NULL,
    organic_residue_count INTEGER NOT NULL,
    pattern_collapse_count INTEGER NOT NULL,
    yield_percent REAL NOT NULL,
    inspection_result TEXT NOT NULL
        CHECK (inspection_result IN ('PASS', 'FAIL', 'HOLD')),
    inspected_at TEXT NOT NULL,

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE
);

-- =========================================================
-- 9. 유지보수 이력
-- =========================================================
CREATE TABLE IF NOT EXISTS maintenance_history (
    maintenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT NOT NULL,
    maintenance_date TEXT NOT NULL,
    component TEXT NOT NULL,
    maintenance_type TEXT NOT NULL
        CHECK (
            maintenance_type IN (
                'INSPECTION',
                'CLEANING',
                'REPAIR',
                'REPLACEMENT',
                'PM'
            )
        ),
    action_description TEXT NOT NULL,
    replacement_flag INTEGER NOT NULL DEFAULT 0
        CHECK (replacement_flag IN (0, 1)),
    usage_hours REAL,
    remaining_life_hours REAL,
    engineer_name TEXT,

    FOREIGN KEY (equipment_id)
        REFERENCES equipment_master(equipment_id)
);

-- =========================================================
-- 10. 이상 감지 결과
-- =========================================================
CREATE TABLE IF NOT EXISTS anomaly_result (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    measured_value REAL,
    reference_value REAL,
    severity TEXT NOT NULL
        CHECK (severity IN ('NORMAL', 'WATCH', 'WARNING', 'CRITICAL')),
    detection_method TEXT NOT NULL,
    message TEXT NOT NULL,

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE,

    FOREIGN KEY (equipment_id)
        REFERENCES equipment_master(equipment_id)
);

-- =========================================================
-- 11. 원인 분석 결과
-- Report에서 원인 순위를 바로 표시하기 위한 테이블
-- =========================================================
CREATE TABLE IF NOT EXISTS root_cause_result (
    cause_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL,
    cause_rank INTEGER NOT NULL,
    cause_name TEXT NOT NULL,
    evidence TEXT NOT NULL,
    contribution_percent REAL NOT NULL,
    confidence_level TEXT NOT NULL
        CHECK (
            confidence_level IN (
                'LOW',
                'MEDIUM',
                'HIGH',
                'VERY_HIGH'
            )
        ),
    analysis_method TEXT NOT NULL,

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE
);

-- =========================================================
-- 12. 조치 제안 결과
-- =========================================================
CREATE TABLE IF NOT EXISTS action_recommendation (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT NOT NULL,
    action_rank INTEGER NOT NULL,
    action_name TEXT NOT NULL,
    target TEXT NOT NULL,
    priority TEXT NOT NULL
        CHECK (priority IN ('LOW', 'MEDIUM', 'HIGH', 'URGENT')),
    expected_effect TEXT NOT NULL,
    responsible_department TEXT NOT NULL,
    action_status TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (action_status IN ('OPEN', 'IN_PROGRESS', 'DONE')),
    created_at TEXT NOT NULL,

    FOREIGN KEY (lot_id)
        REFERENCES lot_history(lot_id)
        ON DELETE CASCADE
);

-- =========================================================
-- 조회 속도 향상을 위한 INDEX
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_lot_equipment
    ON lot_history(equipment_id);

CREATE INDEX IF NOT EXISTS idx_lot_start_time
    ON lot_history(start_time);

CREATE INDEX IF NOT EXISTS idx_sensor_lot
    ON sensor_history(lot_id);

CREATE INDEX IF NOT EXISTS idx_sensor_equipment_time
    ON sensor_history(equipment_id, measured_at);

CREATE INDEX IF NOT EXISTS idx_quality_yield
    ON quality_result(yield_percent);

CREATE INDEX IF NOT EXISTS idx_maintenance_equipment
    ON maintenance_history(equipment_id);

CREATE INDEX IF NOT EXISTS idx_anomaly_lot
    ON anomaly_result(lot_id);

CREATE INDEX IF NOT EXISTS idx_root_cause_lot
    ON root_cause_result(lot_id);

CREATE INDEX IF NOT EXISTS idx_action_lot
    ON action_recommendation(lot_id);