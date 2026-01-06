#!/bin/bash

# interactive_checkpoint_cleaner.sh
# 交互式清理checkpoint文件 - 逐个文件夹确认
# 保留指标最好的N个文件，删除其他的

KEEP_BEST_N=1  # 默认每个文件夹保留最好的3个

echo "=============================================="
echo "  交互式Checkpoint清理工具"
echo "  将逐个文件夹确认，删除指标较差的文件"
echo "=============================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 查找所有包含ckpt文件的目录
echo -e "${BLUE}扫描checkpoint文件...${NC}"
echo ""

# 使用find查找所有.ckpt文件并提取目录
directories=()
while IFS= read -r file; do
    dir=$(dirname "$file")
    if [[ ! " ${directories[@]} " =~ " ${dir} " ]]; then
        directories+=("$dir")
    fi
done < <(find . -name "*.ckpt" -type f)

total_dirs=${#directories[@]}
echo -e "找到 ${GREEN}${total_dirs}${NC} 个包含ckpt文件的目录"
echo ""

# 逐个处理目录
processed=0
for dir in "${directories[@]}"; do
    processed=$((processed + 1))
    
    echo ""
    echo "=============================================="
    echo -e "${BLUE}[${processed}/${total_dirs}] 处理目录:${NC} ${dir}"
    echo "=============================================="
    
    # 切换到目录
    cd "$dir" || continue
    
    # 查找当前目录的ckpt文件
    ckpt_files=()
    for file in *.ckpt; do
        if [[ -f "$file" ]]; then
            ckpt_files+=("$file")
        fi
    done
    
    total_files=${#ckpt_files[@]}
    
    if [[ $total_files -eq 0 ]]; then
        echo -e "${YELLOW}警告: 目录中没有ckpt文件，跳过${NC}"
        cd - > /dev/null
        continue
    fi
    
    echo -e "找到 ${GREEN}${total_files}${NC} 个ckpt文件"
    echo ""
    
    # 按指标类型分组
    score_files=()
    fvd_files=()
    other_files=()
    
    for file in "${ckpt_files[@]}"; do
        if [[ "$file" == *"test_mean_score"* ]]; then
            score_files+=("$file")
        elif [[ "$file" == *"video_fvd"* ]]; then
            fvd_files+=("$file")
        else
            other_files+=("$file")
        fi
    done
    
    # 处理 test_mean_score 文件
    if [[ ${#score_files[@]} -gt 0 ]]; then
        echo "----------------------------------------------"
        echo -e "${BLUE}test_mean_score 文件 (分数越高越好):${NC}"
        echo "----------------------------------------------"
        
        # 按分数排序（降序）
        sorted_scores=()
        while IFS= read -r line; do
            sorted_scores+=("$line")
        done < <(for file in "${score_files[@]}"; do
            # 提取分数
            score=$(echo "$file" | grep -o "test_mean_score=[0-9.]*" | cut -d= -f2)
            # 提取epoch
            epoch=$(echo "$file" | grep -o "epoch=[0-9]*" | cut -d= -f2)
            printf "%s %05d %s\n" "$score" "$epoch" "$file"
        done | sort -rn)
        
        # 显示所有文件
        count=1
        keep_files=()
        delete_files=()
        
        for line in "${sorted_scores[@]}"; do
            score=$(echo "$line" | awk '{print $1}')
            file=$(echo "$line" | awk '{print $3}')
            
            if [[ $count -le $KEEP_BEST_N ]]; then
                echo -e "  ${GREEN}[保留]${NC} $file (score=$score)"
                keep_files+=("$file")
            else
                echo -e "  ${RED}[删除]${NC} $file (score=$score)"
                delete_files+=("$file")
            fi
            count=$((count + 1))
        done
        
        # 确认删除
        if [[ ${#delete_files[@]} -gt 0 ]]; then
            echo ""
            read -p "是否删除上述标记为[删除]的文件? (y/n/s=跳过此组): " -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for file in "${delete_files[@]}"; do
                    if [[ -f "$file" ]]; then
                        rm -v "$file"
                    fi
                done
                echo -e "${GREEN}✓ 已删除 ${#delete_files[@]} 个文件${NC}"
            elif [[ $REPLY =~ ^[Ss]$ ]]; then
                echo -e "${YELLOW}→ 跳过此组文件${NC}"
            else
                echo -e "${YELLOW}→ 取消删除${NC}"
            fi
        else
            echo -e "${YELLOW}没有需要删除的文件${NC}"
        fi
    fi
    
    # 处理 video_fvd 文件
    if [[ ${#fvd_files[@]} -gt 0 ]]; then
        echo ""
        echo "----------------------------------------------"
        echo -e "${BLUE}video_fvd 文件 (FVD越低越好):${NC}"
        echo "----------------------------------------------"
        
        # 按FVD排序（升序）
        sorted_fvds=()
        while IFS= read -r line; do
            sorted_fvds+=("$line")
        done < <(for file in "${fvd_files[@]}"; do
            # 提取FVD
            fvd=$(echo "$file" | grep -o "video_fvd=[0-9.]*" | cut -d= -f2)
            # 提取epoch
            epoch=$(echo "$file" | grep -o "epoch=[0-9]*" | cut -d= -f2)
            printf "%s %05d %s\n" "$fvd" "$epoch" "$file"
        done | sort -n)
        
        # 显示所有文件
        count=1
        keep_files=()
        delete_files=()
        
        for line in "${sorted_fvds[@]}"; do
            fvd=$(echo "$line" | awk '{print $1}')
            file=$(echo "$line" | awk '{print $3}')
            
            if [[ $count -le $KEEP_BEST_N ]]; then
                echo -e "  ${GREEN}[保留]${NC} $file (FVD=$fvd)"
                keep_files+=("$file")
            else
                echo -e "  ${RED}[删除]${NC} $file (FVD=$fvd)"
                delete_files+=("$file")
            fi
            count=$((count + 1))
        done
        
        # 确认删除
        if [[ ${#delete_files[@]} -gt 0 ]]; then
            echo ""
            read -p "是否删除上述标记为[删除]的文件? (y/n/s=跳过此组): " -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for file in "${delete_files[@]}"; do
                    if [[ -f "$file" ]]; then
                        rm -v "$file"
                    fi
                done
                echo -e "${GREEN}✓ 已删除 ${#delete_files[@]} 个文件${NC}"
            elif [[ $REPLY =~ ^[Ss]$ ]]; then
                echo -e "${YELLOW}→ 跳过此组文件${NC}"
            else
                echo -e "${YELLOW}→ 取消删除${NC}"
            fi
        else
            echo -e "${YELLOW}没有需要删除的文件${NC}"
        fi
    fi
    
    # 处理其他文件（如latest.ckpt等）
    if [[ ${#other_files[@]} -gt 0 ]]; then
        echo ""
        echo "----------------------------------------------"
        echo -e "${BLUE}其他ckpt文件:${NC}"
        echo "----------------------------------------------"
        
        for file in "${other_files[@]}"; do
            echo -e "  ${GREEN}[保留]${NC} $file"
        done
        
        # 询问是否删除某些文件
        echo ""
        echo "特殊文件处理:"
        echo "1. latest.ckpt - 通常需要保留"
        echo "2. 孤立的ckpt文件 - 可能需要删除"
        
        for file in "${other_files[@]}"; do
            if [[ "$file" != "latest.ckpt" ]] && [[ ! "$file" =~ ^epoch= ]]; then
                read -p "是否删除 '$file'? (y/n): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -v "$file"
                fi
            fi
        done
    fi
    
    # 返回原始目录
    cd - > /dev/null
    
    # 询问是否跳过剩余目录
    if [[ $processed -lt $total_dirs ]]; then
        echo ""
        echo "----------------------------------------------"
        read -p "继续处理下一个目录? (y/n/q=退出): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Qq]$ ]]; then
            echo "用户退出"
            break
        elif [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过剩余目录"
            break
        fi
    fi
done

echo ""
echo "=============================================="
echo -e "${GREEN}清理完成!${NC}"
echo "=============================================="

# 最后显示磁盘使用情况
echo ""
echo -e "${BLUE}磁盘使用情况:${NC}"
du -sh .
echo ""
echo -e "${BLUE}剩余ckpt文件统计:${NC}"
find . -name "*.ckpt" -type f | wc -l
echo "个ckpt文件"
