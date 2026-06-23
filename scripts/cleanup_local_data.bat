@echo off
echo ==========================================
echo 清理本地训练数据
echo ==========================================
echo.

cd /d e:\taiji

echo [1] 删除原始预训练数据 (15G)...
rmdir /s /q taiji_data\training_data\pretrain 2>nul
echo    完成

echo [2] 删除SFT原始数据 (206M)...
rmdir /s /q taiji_data\training_data\sft 2>nul
echo    完成

echo [3] 删除补充数据 (79M)...
rmdir /s /q taiji_data\training_data\supplementary 2>nul
echo    完成

echo [4] 删除生命体数据 (268K)...
rmdir /s /q taiji_data\training_data\lifeform 2>nul
echo    完成

echo [5] 删除旧合并文件...
del /f /q taiji_data\training_data\sft_merged_clean.jsonl 2>nul
echo    完成

echo [6] 删除下载目录...
rmdir /s /q taiji_data\training_data\downloads 2>nul
rmdir /s /q taiji_data\training_data\downloaded 2>nul
echo    完成

echo [7] 删除其他无用文件...
del /f /q taiji_data\training_data\belle_0.5m.jsonl 2>nul
echo    完成

echo.
echo ==========================================
echo 清理完成！保留的文件：
echo ==========================================
echo.
dir /b taiji_data\training_data\*.jsonl 2>nul
echo.
echo 最终训练集: taiji_data\training_data\pretrain_final.jsonl
echo.

pause
