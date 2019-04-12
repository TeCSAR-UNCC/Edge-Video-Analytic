#!/bin/bash



echo "------------------------- Setting up for Jetson Xavier -------------------------"

function exitIfError {
    if [[ $? -ne 0 ]] ; then
        echo ""
        echo "------------------------- -------------------------"
        echo "Errors detected. Exiting script. The software might have not been successfully installed."
        echo "------------------------- -------------------------"
        exit 1
    fi
}



function executeShInItsFolder {
    # $1 = sh file name
    # $2 = folder where the sh file is
    # $3 = folder to go back
    cd $2
    exitIfError
    sudo chmod +x $1
    exitIfError
    bash ./$1
    exitIfError
    cd $3
    exitIfError
}

executeShInItsFolder "./scripts/ubuntu/install_caffe_and_openpose_JetsonTX2_JetPack3.3.sh" "./" "./"
exitIfError

echo "------------------------- Caffe and OpenPose Installed -------------------------"
echo ""
