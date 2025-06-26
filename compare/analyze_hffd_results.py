# #!/usr/bin/env python3
# """
# Visualise the HFFD benchmark – only `hffd` and `hffd_nadav`.
#
# The script is resilient to CSV files that were produced by older
# versions of the benchmark runner (i.e. without `num_items` /
# `instance_size`).  Missing columns are rebuilt on-the-fly, and plots that
# still cannot be generated are simply skipped with a clear warning.
# """
#
# from pathlib import Path
# import pandas as pd
# import seaborn as sns
# from experiments_csv import single_plot_results
# from matplotlib import pyplot as plt
#
# RESULT_CSV = Path("results/hffd_comparison.csv")
# assert RESULT_CSV.exists(), "run compare_hffd_algorithms.py first"
#
# ALGORITHMS = ["hffd", "hffd_nadav"]
#
#
# # ----------------------------------------------------------------------
# # Helper: add size-related columns if possible
# # ----------------------------------------------------------------------
# def enrich_size_columns(df: pd.DataFrame) -> pd.DataFrame:
#     if "num_items" not in df.columns and {
#         "num_of_agents", "items_per_agent"
#     } <= set(df.columns):
#         df["num_items"] = df["num_of_agents"] * df["items_per_agent"]
#
#     if "instance_size" not in df.columns and {
#         "num_of_agents", "num_items"
#     } <= set(df.columns):
#         df["instance_size"] = df["num_of_agents"] * df["num_items"]
#
#     return df
#
#
# # ----------------------------------------------------------------------
# # 1. Load & filter
# # ----------------------------------------------------------------------
# df = pd.read_csv(RESULT_CSV)
# df = df[df["algorithm"].isin(ALGORITHMS)]
# df = enrich_size_columns(df)
#
# print("\n=== Aggregate results per algorithm (hffd vs hffd_nadav) ===")
# print(
#     df.groupby("algorithm")
#       .agg(success_rate=("success", "mean"),
#            avg_runtime_s=("runtime", "mean"),
#            avg_overload=("max_overload", "mean"))
#       .to_string(float_format=lambda x: f"{x:8.4f}")
# )
#
# # ----------------------------------------------------------------------
# # 2. success-rate  vs  α
# # ----------------------------------------------------------------------
# plt.figure(figsize=(6, 4))
# single_plot_results(
#     RESULT_CSV,
#     filter={"algorithm": ALGORITHMS},
#     x_field="alpha",
#     y_field="success",
#     z_field="algorithm",
#     mean=True,
#     save_to_file="results/success_vs_alpha.png",
# )
# plt.close()
#
# # ----------------------------------------------------------------------
# # 3. runtime  vs  #agents
# # ----------------------------------------------------------------------
# plt.figure(figsize=(6, 4))
# single_plot_results(
#     RESULT_CSV,
#     filter={"algorithm": ALGORITHMS},
#     x_field="num_of_agents",
#     y_field="runtime",
#     z_field="algorithm",
#     mean=True,
#     save_to_file="results/runtime_vs_agents.png",
# )
# plt.close()
#
# # ----------------------------------------------------------------------
# # 4. runtime  vs  instance_size  (only if column now exists)
# # ----------------------------------------------------------------------
# if "instance_size" in df.columns:
#     # Write a temporary CSV *with* the new column for the plotting helper
#     tmp_csv = Path("results/_tmp_with_size.csv")
#     df.to_csv(tmp_csv, index=False)
#
#     plt.figure(figsize=(6, 4))
#     single_plot_results(
#         tmp_csv,
#         filter={"algorithm": ALGORITHMS},
#         x_field="instance_size",
#         y_field="runtime",
#         z_field="algorithm",
#         mean=True,
#         save_to_file="results/runtime_vs_size.png",
#     )
#     plt.close()
#     size_plot_done = True
# else:
#     print("⚠  Skipping runtime_vs_size – cannot deduce `instance_size`.")
#     size_plot_done = False
#
# # ----------------------------------------------------------------------
# # 5. success-rate  vs  noise
# # ----------------------------------------------------------------------
# if "value_noise_ratio" in df.columns:
#     plt.figure(figsize=(6, 4))
#     single_plot_results(
#         RESULT_CSV,
#         filter={"algorithm": ALGORITHMS},
#         x_field="value_noise_ratio",
#         y_field="success",
#         z_field="algorithm",
#         mean=True,
#         save_to_file="results/success_vs_noise.png",
#     )
#     plt.close()
#     noise_plot_done = True
# else:
#     noise_plot_done = False
#     print("⚠  Skipping success_vs_noise – column missing.")
#
# # ----------------------------------------------------------------------
# # 6. Heat-map of max-overload (hffd_nadav only)
# # ----------------------------------------------------------------------
# try:
#     pivot = (df[df.algorithm == "hffd_nadav"]
#              .pivot_table(index="num_of_agents",
#                           columns="alpha",
#                           values="max_overload", aggfunc="mean"))
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
#     plt.title("Mean max-overload (hffd_nadav)")
#     plt.savefig("results/overload_heatmap.png")
#     plt.close()
#     heatmap_done = True
# except Exception as e:
#     heatmap_done = False
#     print(f"⚠  Skipping overload heat-map ({e}).")
#
# # ----------------------------------------------------------------------
# # 7. Summary
# # ----------------------------------------------------------------------
# print("\nSaved figures:")
# print("  • results/success_vs_alpha.png")
# print("  • results/runtime_vs_agents.png")
# if size_plot_done:
#     print("  • results/runtime_vs_size.png")
# if noise_plot_done:
#     print("  • results/success_vs_noise.png")
# if heatmap_done:
#     print("  • results/overload_heatmap.png")
#
# print(
#     "\nIf you prefer the cleaner solution, rerun "
#     "`compare_hffd_algorithms.py` (latest version) which writes "
#     "`instance_size` directly into the CSV.  Then this script will use "
#     "the original file and drop the temporary one."
# )

#!/usr/bin/env python3
"""
Visualise and explain the benchmark of  `hffd`  vs  `hffd_nadav`.

Creates six figures:

  1. success-rate   vs α              (CSV  → single_plot_results)
  2. runtime        vs #agents        (CSV  → single_plot_results)
  3. runtime        vs instance size  (DF   → plot_dataframe)
  4. log-runtime    vs instance size  (DF   → plot_dataframe)
  5. ECDF of runtime                  (DF   → manual)
  6. overload       vs α              (DF   → plot_dataframe)
"""

import pathlib, pandas as pd, numpy as np
from matplotlib import pyplot as plt
from experiments_csv import single_plot_results
from experiments_csv.plot_results import plot_dataframe  # <-- direct DF plotting

RESULT_CSV = pathlib.Path("results/hffd_comparison.csv")
assert RESULT_CSV.exists(), "run compare_hffd_algorithms.py first"

# ---------------------------------------------------------------------
#  0.  load & filter
# ---------------------------------------------------------------------
ALGS = ["hffd", "hffd_nadav"]
df = pd.read_csv(RESULT_CSV)
df = df[df["algorithm"].isin(ALGS)].copy()
df["instance_size"] = df["num_of_agents"] * df["items_per_agent"]

print("\n=== Aggregate results per algorithm (hffd vs hffd_nadav) ===")
print(
    df.groupby("algorithm")
      .agg(success_rate=("success", "mean"),
           avg_runtime_s=("runtime", "mean"),
           avg_overload =("max_overload", "mean"))
      .to_string(float_format=lambda x: f"{x:8.4f}")
)


CSV_CFG = dict(filter={"algorithm": ALGS}, z_field="algorithm", mean=True)

# ----------------------------------------------------------------------
# helper – plot a *DataFrame* (no “filter” arg here!)
# ----------------------------------------------------------------------
def plot_df(data, *, x, y, outfile, y_label=None):
    plt.figure(figsize=(6, 4))
    plot_dataframe(plt, data,
                   x_field=x, y_field=y,
                   z_field="algorithm", mean=True)
    if y_label:
        plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# ----------------------------------------------------------------------
# 1 & 2 – still use the CSV path + filter
# ----------------------------------------------------------------------
single_plot_results(RESULT_CSV,
                    x_field="alpha", y_field="success",
                    save_to_file="results/success_vs_alpha.png",
                    **CSV_CFG)

single_plot_results(RESULT_CSV,
                    x_field="num_of_agents", y_field="runtime",
                    save_to_file="results/runtime_vs_agents.png",
                    **CSV_CFG)

# ----------------------------------------------------------------------
# 3. runtime vs instance size
# ----------------------------------------------------------------------
plot_df(df, x="instance_size", y="runtime",
        outfile="results/runtime_vs_size.png")

# 4. log-runtime vs instance size
df["log_runtime"] = np.log10(df["runtime"] + 1e-9)
plot_df(df, x="instance_size", y="log_runtime",
        outfile="results/log_runtime_vs_size.png",
        y_label="log10 runtime (s)")

# 5. ECDF of runtime
plt.figure(figsize=(6,4))
for alg in ALGS:
    r = np.sort(df[df["algorithm"] == alg]["runtime"])
    y = np.linspace(0, 1, len(r), endpoint=False)
    plt.step(r, y, where="post", label=alg)
plt.xlabel("runtime (s)")
plt.ylabel("empirical CDF")
plt.legend()
plt.tight_layout()
plt.savefig("results/runtime_ecdf.png")
plt.close()

# 6. overload vs α
plot_df(df, x="alpha", y="max_overload",
        outfile="results/overload_vs_alpha.png")

print("\nSaved figures:")
for f in ["success_vs_alpha.png",
          "runtime_vs_agents.png",
          "runtime_vs_size.png",
          "log_runtime_vs_size.png",
          "runtime_ecdf.png",
          "overload_vs_alpha.png"]:
    print(f"  • results/{f}")