 # CompBench Evaluation Guide

   CompBench is a comprehensive benchmark dataset for evaluating image editing model performance. This guide will help you quickly get started with using CompBench for model evaluation.

   ## Quick Start

   ### 1. Dataset Download and Setup

   First, download the CompBench dataset in parquet format from Hugging Face:

   Visit: https://huggingface.co/datasets/BohanJia/CompBench/viewer/default/train?p=17&views%5B%5D=train

   After downloading, move the files to your working directory:

   ```bash
   cd /CompBench
   ```

   ### 2. Data Extraction

   Use the provided script to extract the parquet files:

   ```bash
   python extract_parquet.py --input_dir ./data --output_dir your_dir
   ```

   **Parameters:**

   - `--input_dir`: Directory containing the parquet files
   - `--output_dir`: Output directory for extracted files

   ### 3. Prepare Test Files

   ⚠️ **Important:** Your edited files must maintain the same directory structure as the `edited_image` folders under each task in the extracted `tasks` folder.

   Ensure your file structure follows this pattern:

   ```
   your_edited_dir/
   ├── action/
   ├── add/
   ├── implicit_reasoning/
   ├── location/
   ├── multi_object_remove/
   ├── multi_turn_add/
   ├── multi_turn_remove/
   ├── remove/
   ├── replace/
   └── view/
   ```

   ### 4. Run Evaluation

   Use the corresponding test scripts for evaluation. For example, for implicit editing tasks:

   ```bash
   python eval_implicit.py --edited_dir your_edited_dir
   ```

   **Parameters:**

   - `--edited_dir`: Directory containing your prepared edited images

   ### 5. Multi-turn Editing Special Processing

   For multi-turn editing tasks, an additional preprocessing step is required before testing:

   ```bash
   python convert_multi_turn.py
   ```

   This script merges the metadata from `multi_turn_add` and `multi_turn_remove` tasks to ensure proper multi-turn editing evaluation.

   ## Directory Structure

   The complete project directory structure should look like this:

   ```
   CompBench_dataset/
   ├── .cache/                  # Cache directory
   ├── data/                    # Original parquet files
   ├── tasks/                   # Extracted task data
   │   ├── action/
   │   ├── add/
   │   ├── implicit_reasoning/
   │   ├── location/
   │   ├── multi_object_remove/
   │   ├── multi_turn_add/
   │   ├── multi_turn_remove/
   │   ├── remove/
   │   ├── replace/
   │   └── view/
   ├── convert_multi_turn.py   # Multi-turn editing preprocessing script
   ├── eval_implicit.py        # Implicit reasoning evaluation script
   ├── eval_local_clip_img.py  # Local CLIP image evaluation script
   ├── eval_local_editing.py   # Local editing evaluation script
   ├── eval_multi_clip_img.py  # Multi CLIP image evaluation script
   ├── eval_multi_editing.py   # Multi editing evaluation script
   ├── extract_parquet.py      # Data extraction script
   ```

   ## Available Evaluation Scripts

   CompBench provides multiple evaluation scripts for different types of editing tasks:

   - `eval_implicit.py` - For implicit reasoning tasks
   - `eval_local_clip_img.py` - For local CLIP-based image evaluation
   - `eval_local_editing.py` - For local editing evaluation
   - `eval_multi_clip_img.py` - For multi-image CLIP evaluation
   - `eval_multi_editing.py` - For multi-image editing evaluation
   - `compare_images.py` - Utility for comparing images

   ## Usage Examples

   ### Basic Evaluation

   ```bash
   # Extract data
   python extract_parquet.py --input_dir ./data --output_dir ./tasks
   
   # Run implicit reasoning evaluation
   python eval_implicit.py --edited_dir ./your_edited_dir
   
   # Run local editing evaluation
   python eval_local_editing.py --edited_dir ./your_edited_dir
   
   # Run CLIP-based evaluations
   python eval_local_clip_img.py --edited_dir ./your_edited_dir
   python eval_multi_clip_img.py --edited_dir ./your_edited_dir
   ```

   ### Multi-turn Editing Evaluation

   ```bash
   # Preprocess multi-turn data
   python convert_multi_turn.py
   
   # Run multi-turn editing evaluation
   python eval_multi_editing.py --edited_dir ./your_multi_turn_images
   ```

   ## Tips and Best Practices

   1. **File Structure Consistency**: Always ensure your edited images follow the exact same directory structure as the original dataset
   2. **Image Formats**: Verify that your edited images are in the correct format (typically JPEG or PNG)
   3. **Naming Convention**: Keep the same file names as in the original dataset
   4. **Quality Check**: Validate your edited images before running evaluation to avoid processing errors

   
