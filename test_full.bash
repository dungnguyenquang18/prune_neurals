

nohup sh -c 'python test_prune_neurals.py -method base > logs/base.log 2>&1 && \
python test_prune_neurals.py -method kmeans > logs/kmeans.log 2>&1 && \
python test_prune_neurals.py -method distance-based-clustering > logs/distance-based-clustering.log 2>&1 && \
python test_prune_neurals.py -method kmedoids > logs/kmedoids.log 2>&1' &
git add .
git commit -m "Run pruning experiments and save results to logs"
git push origin dungnq