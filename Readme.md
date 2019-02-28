# Machine Learning

# Data Cleaning
First of all we merge all the data after extracting the **NewData.zip** file and creating data files for 'Stopper, Wall and Wall2'

The different files in the seperate folder **Stopper**, **Wall**, **Wall2** are depending upon the distance like 60cm, 70cm,.... 300cm

# Data Analysis
After getting the data, we need to apply FFT and cut it with the frequency ranges **30MHz to 500MHz** so that the data is according to the requirement(as the device which we collect functions at 40Mhz)

# Applying AI
We train the **MLP** with data frame Stopper and Wall using the same distance data.

And for prediction we may use Stopper or Wall files from different distances.
