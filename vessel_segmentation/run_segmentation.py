# Save deconvolved image for debugging
sitk.WriteImage(
    sitk.GetImageFromArray(deconv_image),
    os.path.join(output_dirs['intermediate'], 'deconvolved_image.nrrd')
)

# Calculate vesselness using deconvolved image and eroded mask
vesselness_results = calculate_vesselness(
    deconv_image,
    eroded_mask,
    scales,
    output_dir=output_dirs['intermediate']
) 