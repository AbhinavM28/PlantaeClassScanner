# Launch model training with logging

echo "🚀 Starting PlantaeClassScanner model training in background..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs/training

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training/training_${TIMESTAMP}.log"

echo " Training log: $LOG_FILE"
echo " Monitor with: tail -f $LOG_FILE"
echo " Or view TensorBoard: tensorboard --logdir models/training"
echo ""

# Launch training in background
nohup python src/training/train_model.py \
    --epochs 50 \
    --batch-size 32 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✅ Training started (PID: $TRAIN_PID)"
echo "⏱️  Estimated time: 30-60 minutes"
echo ""
echo "Commands:"
echo "  Monitor progress:  tail -f $LOG_FILE"
echo "  Check if running:  ps aux | grep $TRAIN_PID"
echo "  Stop training:     kill $TRAIN_PID"
echo ""