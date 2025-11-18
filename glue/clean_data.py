import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.ml.feature import StringIndexer

# Job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_BUCKET', 'OUTPUT_BUCKET'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# S3 paths
input_csv_path = f"s3://{args['INPUT_BUCKET']}/raw/"
input_images_path = f"s3://{args['INPUT_BUCKET']}/raw/img/"
output_csv_path = f"s3://{args['OUTPUT_BUCKET']}/cleaned/"
output_images_path = f"s3://{args['OUTPUT_BUCKET']}/cleaned/img/"

s3_client = boto3.client('s3')

print(f"Reading data from: {input_csv_path}")

# Read CSV data
df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_csv_path)

print(f"Initial record count: {df.count()}")
print("Initial schema:")
df.printSchema()

# Data cleaning steps
print("Cleaning data...")

# 1. Remove duplicates based on image_id (should be unique)
df = df.dropDuplicates(['image_id'])

# 2. Remove rows with null values in critical columns
df = df.filter(
    col('image_id').isNotNull() & 
    col('dx').isNotNull() & 
    (col('image_id') != '')
)

# 3. Handle missing values in optional columns
# Fill missing age with median
median_age = df.approxQuantile('age', [0.5], 0.01)[0]
df = df.withColumn('age', when(col('age').isNull(), median_age).otherwise(col('age')))

# Fill missing sex with 'unknown'
df = df.withColumn('sex', when(col('sex').isNull(), 'unknown').otherwise(col('sex')))

# Fill missing localization with 'unknown'
df = df.withColumn('localization', when(col('localization').isNull(), 'unknown').otherwise(col('localization')))

# Fill missing dx_type with 'unknown'
df = df.withColumn('dx_type', when(col('dx_type').isNull(), 'unknown').otherwise(col('dx_type')))

# 4. Apply StringIndexer for encoding dx (target variable)
print("Encoding target variable 'dx'...")
indexer = StringIndexer(inputCol="dx", outputCol="dx_label")
indexer_model = indexer.fit(df)
df = indexer_model.transform(df)

# Get label mapping
label_mapping = {label: idx for idx, label in enumerate(indexer_model.labels)}
print(f"Label mapping: {label_mapping}")

# 5. Optional: Encode categorical features (sex, localization, dx_type)
sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_encoded")
sex_model = sex_indexer.fit(df)
df = sex_model.transform(df)

localization_indexer = StringIndexer(inputCol="localization", outputCol="localization_encoded")
localization_model = localization_indexer.fit(df)
df = localization_model.transform(df)

dx_type_indexer = StringIndexer(inputCol="dx_type", outputCol="dx_type_encoded")
dx_type_model = dx_type_indexer.fit(df)
df = dx_type_model.transform(df)

print(f"Cleaned record count: {df.count()}")
print("Cleaned schema:")
df.printSchema()

# Create image label mapping for organizing images
image_label_map = {row['image_id']: int(row['dx_label']) for row in df.select('image_id', 'dx_label').collect()}
print(f"Created label mapping for {len(image_label_map)} images")

# Create directories for each label in S3
for label_idx in range(len(indexer_model.labels)):
    s3_client.put_object(
        Bucket=args['OUTPUT_BUCKET'],
        Key=f"cleaned/img/{label_idx}/"
    )
    print(f"Created directory for label {label_idx}: {indexer_model.labels[label_idx]}")

# Process and organize images by label
print("Processing and organizing images...")
paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=args['INPUT_BUCKET'], Prefix='raw/img/')

images_processed = 0
images_skipped = 0

for page in pages:
    if 'Contents' not in page:
        continue
    
    for obj in page['Contents']:
        key = obj['Key']
        
        # Skip directories
        if key.endswith('/'):
            continue
        
        # Skip if not in img folder
        if '/img/' not in key:
            continue
        
        # Get image name and ID
        image_name = key.split('/')[-1]
        
        # Skip non-image files
        if not (image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))):
            continue
            
        image_id = image_name.split('.')[0]
        
        try:
            # Get label for image
            if image_id in image_label_map:
                img_label = image_label_map[image_id]
                
                # Copy to new location organized by label
                copy_source = {
                    'Bucket': args['INPUT_BUCKET'],
                    'Key': key
                }
                dest_key = f"cleaned/img/{img_label}/{image_name}"
                
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=args['OUTPUT_BUCKET'],
                    Key=dest_key
                )
                images_processed += 1
                
                if images_processed % 100 == 0:
                    print(f"Processed {images_processed} images...")
            else:
                print(f"Warning: No label found for image {image_id}")
                images_skipped += 1
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            images_skipped += 1

print(f"Image processing complete: {images_processed} processed, {images_skipped} skipped")

# Write cleaned CSV data to S3
print(f"Writing cleaned CSV data to: {output_csv_path}")
df.write.mode('overwrite').option("header", "true").csv(output_csv_path)

# Also save as a single CSV file for easier use
df.coalesce(1).write.mode('overwrite').option("header", "true").csv(f"{output_csv_path}single/")

# Save label mappings as a separate file
label_mapping_df = spark.createDataFrame(
    [(label, idx) for label, idx in label_mapping.items()],
    ['dx_name', 'dx_label']
)
label_mapping_df.coalesce(1).write.mode('overwrite').option("header", "true").csv(f"{output_csv_path}label_mapping/")

print("Data cleaning and organization complete!")

# Delete raw data
print("Deleting raw data...")
try:
    delete_paginator = s3_client.get_paginator('list_objects_v2')
    delete_pages = delete_paginator.paginate(Bucket=args['INPUT_BUCKET'], Prefix='raw/')
    
    total_deleted = 0
    for page in delete_pages:
        if 'Contents' in page:
            objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
            
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=args['INPUT_BUCKET'],
                    Delete={'Objects': objects_to_delete}
                )
                total_deleted += len(objects_to_delete)
                print(f"Deleted {len(objects_to_delete)} objects")
    
    print(f"Data deletion complete! Total deleted: {total_deleted}")
except Exception as e:
    print(f"Error deleting data: {e}")

print("Job completed successfully!")

job.commit()