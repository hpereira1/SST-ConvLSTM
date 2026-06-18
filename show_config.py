"""
Script para mostrar a configuração atual do projeto
Útil para verificar quais valores serão usados no treinamento
"""

from src.utils.config import Config


def main():
    """Display current configuration"""
    config = Config()
    
    print("=" * 70)
    print("CURRENT CONFIGURATION (from config.py)")
    print("=" * 70)
    
    print("\n📊 DATA CONFIGURATION")
    print(f"  Data path:        {config.data.nc_path}")
    print(f"  Lookback:         {config.data.lookback} timesteps")
    print(f"  Horizons:         {config.data.horizons}")
    print(f"  Min valid:        {config.data.min_valid}")
    print(f"  Train period:     {config.data.train_slice[0]} to {config.data.train_slice[1]}")
    print(f"  Val period:       {config.data.val_slice[0]} to {config.data.val_slice[1]}")
    print(f"  Test period:      {config.data.test_slice[0]} to {config.data.test_slice[1]}")
    
    print("\n🏗️  MODEL CONFIGURATION")
    print(f"  Base filters:     {config.model.base_filters}")
    print(f"  Kernel size:      {config.model.conv_kernel_size}")
    
    print("\n🚀 TRAINING CONFIGURATION")
    print(f"  Batch size:       {config.training.batch_size}")
    print(f"  Epochs:           {config.training.epochs}")
    print(f"  Learning rate:    {config.training.learning_rate}")
    print(f"  Clip norm:        {config.training.clipnorm}")
    print(f"  Mixed precision:  {config.training.use_mixed_precision}")
    print(f"  Memory growth:    {config.training.enable_memory_growth}")

    print("\n💾 OUTPUT PATHS")
    print(f"  Model save:       {config.model_save_path}")
    print(f"  Checkpoints:      {config.checkpoint_dir}")
    print(f"  TensorBoard:      {config.tensorboard_dir}")
    
    print("\n" + "=" * 70)
    print("To override any value, use CLI arguments:")
    print("  python train.py --batch-size 8 --base-filters 24")
    print("\nTo see all CLI options:")
    print("  python train.py --help")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

