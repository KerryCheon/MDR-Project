from Modeling.Src.soilmoist_fl.cli import DEFAULT_CONFIG_PATH, DEFAULT_RUNS_DIR, run_feature_selection


def main():
    run_feature_selection(
        config_path=DEFAULT_CONFIG_PATH,
        base_runs_dir=DEFAULT_RUNS_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
