# picasso

A simple python implementation of the below paper for the linear unmixing of spectrally overlapping signal.

[PICASSO allows ultra-multiplexed fluorescence imaging of spatially overlapping proteins without reference spectra measurements](https://www.nature.com/articles/s41467-022-30168-z)

## Notes
Images to be unmixed should be smoothed or downsampled to prevent poor performance. Right now the inputs to the functions are assuming CYX format, with the unmixing occurring along the C axis, but this can be extended to specify an arbitrary axis.