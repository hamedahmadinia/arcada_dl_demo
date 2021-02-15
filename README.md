## Demo

The code in this repository is based on [CSC's Introduction to deep learning](https://github.com/csc-training/intro-to-dl/tree/master/day2) exercises.

### Setup

1. Login to Puhti using your csc account:

        ssh -l yourcscusername puhti.csc.fi
        
2. Clone and cd to the demo repository:

        git clone https://github.com/nyholmju/arcada_dl_demo.git demo
        cd demo

### How to submit jobs

1. Submit a job to the slurm queue:

        sbatch <slurm_script> <training_script>

   You can also specify additional command line arguments to the training script, e.g.

        sbatch run-tf2.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

2. See the status of your jobs or the queue you are using:

        squeue -l -u yourcscusername
        squeue -l -p gputest

3. You can examine the output of the training script while the job is running:

        tail -f slurm-xxxxxxxx.out

4. After the job has finished, check batch jobs runtime and resource utilization using `seff`

        seff slurmjobid # The slurm job id is the xxxxxxxx part in the file name (slurm-xxxxxxxx.out)


### TF2/Keras - Image classification: dogs vs. cats

* *tf2-dvc-cnn-simple.py*: Dogs vs. cats with a CNN trained from scratch
* *tf2-dvc-cnn-pretrained.py*: Dogs vs. cats with a pre-trained CNN
* *tf2-dvc-cnn-evaluate.py*: Evaluate a trained CNN with test data
* *tf2-dvc-cnn-predict.py*: Use trained CNN to predict the class label of supplied jpg file

To train a simple CNN from scratch:

        sbatch run-tf2.sh tf2-dvc-cnn-simple.py

To train a CNN using transfer learning:

         sbatch run-tf2.sh tf2-dvc-cnn-pretrained.py

To evaluate trained model on the test set, append models file name as a command line argument, e.g.

        sbatch run-tf2.sh tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5

To predict the class label of a supplied jpg file, append models file name and jpg file name as command line arguments, e.g.

        sbatch run-tf2.sh tf2-dvc-cnn-predict.py dvc-vgg16-finetune.h5 image.jpg
