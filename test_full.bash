#!/bin/bash

# Tạo thư mục logs nếu chưa có
# mkdir -p logs

# Chạy tuần tự các phương pháp pruning
# Sử dụng && để đảm bảo nếu một bước lỗi nặng thì có thể dừng lại (hoặc dùng ; nếu muốn chạy bất chấp)
echo "Starting pruning experiments..."

python test_prune_neurals.py -method base > logs/base.log 2>&1 && \
python test_prune_neurals.py -method kmeans > logs/kmeans.log 2>&1 && \
python test_prune_neurals.py -method distance-based-clustering > logs/distance-based-clustering.log 2>&1 && \
python test_prune_neurals.py -method kmedoids > logs/kmedoids.log 2>&1

# Sau khi tất cả các lệnh trên chạy xong, tiến hành Git
echo "Experiments finished. Pushing results to Git..."

git add .
git commit -m "Run pruning experiments and save results to logs single thread"
git push origin dungnq_single_thread