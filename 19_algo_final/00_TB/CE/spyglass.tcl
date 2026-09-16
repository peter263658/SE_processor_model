# =================================================================
# SpyGlass Run Script
# =================================================================

# 1. 初始化 Project
# -----------------------------------------------------------------
new_project my_design_check -force
# -force 會覆蓋掉舊的專案設定，避免衝突

# 2. 設定參數 (Import Search Path, Pre-define macros)
# -----------------------------------------------------------------

# set_option define_macro "SYNTHESIS" ;# 如果需要定義 macro

# 3. 讀入設計檔案 (Verilog / SystemVerilog)
# -----------------------------------------------------------------
# 依序讀入你的檔案，或者讀取一個 filelist
read_file -type systemverilog {
    ../../01_RTL/Define/Param_def.sv
    ../../01_RTL/Define/Mem_def.sv
    ../../01_RTL/Designware/DW_fp_add_inst.sv
    ../../01_RTL/Designware/DW_fp_addsub_inst.sv
    ../../01_RTL/Designware/DW_fp_mult_inst.sv
    ../../01_RTL/CE/CE_AT.sv
}

# 4. 設定 Top Module
# -----------------------------------------------------------------
set_option top CE_AT  ;# 請改成你實際的 Top Module 名稱

# 5. 設定檢查目標 (Goals)
# -----------------------------------------------------------------
# 一般 Linting 檢查 (語法、Synthesizability)
current_goal lint/lint_rtl

# 如果你要跑 CDC (Clock Domain Crossing)，通常會換成 cdc/cdc_verify_struct
# current_goal cdc/cdc_verify_struct

# 6. 開始執行檢查
# -----------------------------------------------------------------
run_goal

# 7. 產生報告並儲存
# -----------------------------------------------------------------
# 報告通常會自動產生在工作目錄下，但確認 save 是一個好習慣
save_project
exit