#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <chrono>

#include "IOWrapper/ImageDisplay.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"


#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

#include <opencv/cv.hpp>
#include <opencv/highgui.h>



std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
double rescale = 1;
bool reverse_dso = false;

bool disableROS = false;

int start=330;
int end_dso=800;

bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;

bool readSemantics = false;
int numSemanticsClass = 19;

int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
  printf("Caught signal %d\n",s);
  exit(1);
}

void exitThread()
{
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_exit_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  firstRosSpin=true;
  while(true) pause();
}

void settingsDefault(int preset)
{
  printf("\n=============== PRESET Settings: ===============\n");
  if(preset == 0 || preset == 1)
    {
      printf("DEFAULT settings:\n"
             "- %s real-time enforcing\n"
             "- 2000 active points\n"
             "- 5-7 active frames\n"
             "- 1-6 LM iteration each KF\n"
             "- original image resolution\n", preset==0 ? "no " : "1x");

      playbackSpeed = (preset==0 ? 0 : 1);
      preload = preset==1;

      setting_desiredImmatureDensity = 1500;    //original 1500. set higher
      setting_desiredPointDensity = 2000;       //original 2000
      setting_minFrames = 5;
      setting_maxFrames = 7;
      setting_maxOptIterations=6;
      setting_minOptIterations=1;

      setting_kfGlobalWeight=0.3;   // original is 1.0. 0.3 is a balance between speed and accuracy. if tracking lost, set this para higher
      setting_maxShiftWeightT= 0.04f * (640 + 128);   // original is 0.04f * (640+480); this para is depend on the crop size.
      setting_maxShiftWeightR= 0.04f * (640 + 128);    // original is 0.0f * (640+480);
      setting_maxShiftWeightRT= 0.02f * (640 + 128);  // original is 0.02f * (640+480);

      setting_logStuff = false;
    }

  if(preset == 2 || preset == 3)
    {
      printf("FAST settings:\n"
             "- %s real-time enforcing\n"
             "- 800 active points\n"
             "- 4-6 active frames\n"
             "- 1-4 LM iteration each KF\n"
             "- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

      playbackSpeed = (preset==2 ? 0 : 5);
      preload = preset==3;
      setting_desiredImmatureDensity = 600;
      setting_desiredPointDensity = 800;
      setting_minFrames = 4;
      setting_maxFrames = 6;
      setting_maxOptIterations=4;
      setting_minOptIterations=1;

      benchmarkSetting_width = 424;
      benchmarkSetting_height = 320;

      setting_logStuff = false;
    }

  printf("==============================================\n");
}

void parseArgument(char* arg)
{
  int option;
  float foption;
  char buf[1000];


  if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
      if(option==1)
        {
          useSampleOutput = true;
          printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
      return;
    }

  if(1==sscanf(arg,"quiet=%d",&option))
    {
      if(option==1)
        {
          setting_debugout_runquiet = true;
          printf("QUIET MODE, I'll shut up!\n");
        }
      return;
    }

  if(1==sscanf(arg,"preset=%d",&option))
    {
      settingsDefault(option);
      return;
    }


  if(1==sscanf(arg,"rec=%d",&option))
    {
      if(option==0)
        {
          disableReconfigure = true;
          printf("DISABLE RECONFIGURE!\n");
        }
      return;
    }


  if(1==sscanf(arg,"noros=%d",&option))
    {
      if(option==1)
        {
          disableROS = true;
          disableReconfigure = true;
          printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
      return;
    }

  if(1==sscanf(arg,"nolog=%d",&option))
    {
      if(option==1)
        {
          setting_logStuff = false;
          printf("DISABLE LOGGING!\n");
        }
      return;
    }
  if(1==sscanf(arg,"reverse=%d",&option))
    {
      if(option==1)
        {
          reverse_dso = true;
          printf("REVERSE!\n");
        }
      return;
    }
  if(1==sscanf(arg,"nogui=%d",&option))
    {
      if(option==1)
        {
          disableAllDisplay = true;
          printf("NO GUI!\n");
        }
      return;
    }
  if(1==sscanf(arg,"nomt=%d",&option))
    {
      if(option==1)
        {
          multiThreading = false;
          printf("NO MultiThreading!\n");
        }
      return;
    }
  if(1==sscanf(arg,"prefetch=%d",&option))
    {
      if(option==1)
        {
          prefetch = true;
          printf("PREFETCH!\n");
        }
      return;
    }
  if(1==sscanf(arg,"start=%d",&option))
    {
      start = option;
      printf("START AT %d!\n",start);
      return;
    }
  if(1==sscanf(arg,"end=%d",&option))
   { 
      end_dso = option;
      printf("END AT %d!\n",end_dso);
      return;
    }

  if(1==sscanf(arg,"files=%s",buf))
    {
      source = buf;
      printf("loading data from %s!\n", source.c_str());
      return;
    }

  if(1==sscanf(arg,"calib=%s",buf))
    {
      calib = buf;
      printf("loading calibration from %s!\n", calib.c_str());
      return;
    }

  if(1==sscanf(arg,"vignette=%s",buf))
    {
      vignette = buf;
      printf("loading vignette from %s!\n", vignette.c_str());
      return;
    }

  if(1==sscanf(arg,"gamma=%s",buf))
    {
      gammaCalib = buf;
      printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
      return;
    }

  if(1==sscanf(arg,"rescale=%f",&foption))
    {
      rescale = foption;
      printf("RESCALE %f!\n", rescale);
      return;
    }

  if(1==sscanf(arg,"speed=%f",&foption))
    {
      playbackSpeed = foption;
      printf("PLAYBACK SPEED %f!\n", playbackSpeed);
      return;
    }

  if(1==sscanf(arg,"save=%d",&option))
    {
      if(option==1)
        {
          debugSaveImages = true;
          if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
          if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
          if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
          if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
          printf("SAVE IMAGES!\n");
        }
      return;
    }

  if(1==sscanf(arg,"mode=%d",&option))
    {
      mode = option;
      if(option==0)
        {
          printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
      if(option==1)
        {
          printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
          setting_photometricCalibration = 0;
          setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
          setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
      if(option==2)
        {
          printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
          setting_photometricCalibration = 0;
          setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
          setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
          setting_minGradHistAdd=3;
        }
      return;
    }

  printf("could not parse argument \"%s\"!!!!\n", arg);
}


int main(int argc, char ** argv) {

  for(int i=1; i<argc;i++)
    parseArgument(argv[i]);


  ImageFolderReader::DataType dType;
  dType.readColor = true;
  dType.readGray = true;
  dType.readSemantics = readSemantics;
  dType.numSemanticsClass = numSemanticsClass;
  // only read semantics at the left lens
  ImageFolderReader* reader = new ImageFolderReader(source+"/image_2", calib, gammaCalib, vignette, dType );
  dType.readSemantics = false; // do not read semantics on the right camera
  ImageFolderReader* reader_right = new ImageFolderReader(source+"/image_3", calib, gammaCalib, vignette, dType);
  reader->setGlobalCalibration();
  reader_right->setGlobalCalibration();

  FullSystem fullsystem;
  fullsystem.setGammaFunction(reader->getPhotometricGamma());
  fullsystem.linearizeOperation = (playbackSpeed==0);
	

  int num_img = reader->getNumImages();
  
  int i = 561;
  while (i < num_img) {
    ImageAndExposure * image = reader->getImage(i);
    ImageAndExposure * image_right = reader_right->getImage(i);
                                                              
    std::cout<<"\n\n======================================================\n";
    std::cout<<"[main] New frame # "<<i<<std::endl;

    fullsystem.recordPcds(image, image_right, i);


    delete image;
    delete image_right;
    i++;
  }


  delete reader;
  delete reader_right;

  return 0;
  
}
