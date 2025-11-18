import os
import json
import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'cancer-pred-pipeline-data')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'cancer-pred-pipeline-models')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

s3_client = boto3.client('s3', region_name=AWS_REGION)

def download_data(bucket, prefix='cleaned/', local_dir='data/'):
    """Download data from S3"""
    paginator = s3_client.get_paginator('list_objects_v2')

    print(f"Downloading data from s3://{bucket}/{prefix}")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']

            if key.endswith('/'):
                continue

            local_path = os.path.join(local_dir, key.replace(prefix, ''))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            s3_client.download_file(bucket, key, local_path)

    print(f"Download complete to {local_dir}")

def load_and_merge_csv(csv_dir):
    """Load CSV metadata from the cleaned directory"""
    print(f"Loading CSV metadata from {csv_dir}")
    
    # Find all CSV files in the directory
    csv_files = []
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("Warning: No CSV files found")
        return None
    
    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        return None
    
    metadata_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded metadata for {len(metadata_df)} images")
    print(f"Columns: {metadata_df.columns.tolist()}")
    print(f"Sample data:\n{metadata_df.head()}")
    
    return metadata_df

def prepare_data_with_metadata(data_dir, csv_dir, img_height=224, img_width=224, batch_size=32, validation_split=0.2):
    """Prepare data generators with metadata integration"""
    
    # Load metadata
    metadata_df = load_and_merge_csv(csv_dir)
    
    # Image augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Found {train_generator.samples} training images")
    print(f"Found {val_generator.samples} validation images")
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, num_classes, metadata_df

def build_model(num_classes, img_height=224, img_width=224):
    """Build the CNN model"""
    print("Building model...")
    
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Model architecture
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print("Model built successfully")
    model.summary()
    
    return model

def train_model(model, train_generator, val_generator, epochs=10):
    """Train the model"""
    print(f"Training model for {epochs} epochs...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("Training complete")
    return history

def evaluate_model(model, val_generator):
    """Evaluate the model and calculate metrics"""
    print("Evaluating model...")
    
    # Reset generator
    val_generator.reset()
    
    # Get predictions
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_probs = np.max(predictions, axis=1)
    
    # Get actual labels
    true_classes = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, 
        predicted_classes, 
        average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_classes, 
        predicted_classes, 
        average=None,
        labels=list(range(len(class_labels)))
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_labels,
        output_dict=True
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_labels': class_labels,
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support.tolist()
        }
    }
    
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nPer-Class Metrics:")
    for i, label in enumerate(class_labels):
        print(f"  {label}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall: {recall_per_class[i]:.4f}")
        print(f"    F1 Score: {f1_per_class[i]:.4f}")
        print(f"    Support: {support[i]}")
    print(f"{'='*50}\n")
    
    return metrics

def save_model_and_artifacts(model, metrics, class_mapping, metadata_df, output_dir):
    
    print(f"Saving model and artifacts to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save class mappings
    mapping_path = os.path.join(output_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")
    
    # Save metadata summary
    if metadata_df is not None:
        metadata_summary = {
            'total_samples': len(metadata_df),
            'columns': metadata_df.columns.tolist(),
            'class_distribution': metadata_df['dx'].value_counts().to_dict() if 'dx' in metadata_df.columns else {},
            'age_statistics': {
                'mean': float(metadata_df['age'].mean()) if 'age' in metadata_df.columns else None,
                'std': float(metadata_df['age'].std()) if 'age' in metadata_df.columns else None,
                'min': float(metadata_df['age'].min()) if 'age' in metadata_df.columns else None,
                'max': float(metadata_df['age'].max()) if 'age' in metadata_df.columns else None
            } if 'age' in metadata_df.columns else {},
            'sex_distribution': metadata_df['sex'].value_counts().to_dict() if 'sex' in metadata_df.columns else {},
            'localization_distribution': metadata_df['localization'].value_counts().to_dict() if 'localization' in metadata_df.columns else {}
        }
        
        metadata_summary_path = os.path.join(output_dir, 'metadata_summary.json')
        with open(metadata_summary_path, 'w') as f:
            json.dump(metadata_summary, f, indent=2)
        print(f"Metadata summary saved to {metadata_summary_path}")

def upload_to_s3(local_dir, bucket, prefix):
    """Upload artifacts to S3"""
    s3_client = boto3.client('s3')
    
    print(f"Uploading artifacts to s3://{bucket}/{prefix}")
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, relative_path).replace('\\', '/')
            s3_client.upload_file(local_path, bucket, s3_key)
            print(f"Uploaded {relative_path}")
    
    print("Upload complete")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-height', type=int, default=224)
    parser.add_argument('--img-width', type=int, default=224)
    parser.add_argument('--validation-split', type=float, default=0.2)
    
    # CSV metadata directory (relative to train directory)
    parser.add_argument('--csv-dir', type=str, default='../csv', 
                        help='Directory containing CSV metadata files')
    
    args = parser.parse_args()
    
    # Determine CSV directory path
    # If running in SageMaker, CSV should be in a separate channel or within train data
    # Adjust this path based on your S3 structure
    csv_dir = os.path.join(os.path.dirname(args.train), 'csv') if args.csv_dir == '../csv' else args.csv_dir
    
    # If CSV dir doesn't exist, try looking in train directory
    if not os.path.exists(csv_dir):
        csv_dir = os.path.join(args.train, '..', 'csv')
    
    print(f"Training images directory: {args.train}")
    print(f"CSV metadata directory: {csv_dir}")
    
    # Prepare data with metadata
    train_generator, val_generator, num_classes, metadata_df = prepare_data_with_metadata(
        args.train,
        csv_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Build model
    model = build_model(
        num_classes,
        img_height=args.img_height,
        img_width=args.img_width
    )
    
    # Train model
    history = train_model(model, train_generator, val_generator, epochs=args.epochs)
    
    # Evaluate model
    metrics = evaluate_model(model, val_generator)
    
    # Save model and artifacts
    save_model_and_artifacts(
        model,
        metrics,
        train_generator.class_indices,
        metadata_df,
        args.model_dir
    )
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model saved to: {args.model_dir}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()