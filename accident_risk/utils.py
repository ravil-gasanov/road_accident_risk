def make_experiment_name(filename: str) -> str:
    return filename.split("\\")[-1].replace(".py", "")
