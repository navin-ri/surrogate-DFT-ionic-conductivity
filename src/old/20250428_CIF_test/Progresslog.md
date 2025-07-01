# Progress log
## Current version
- parsed cif files are used to create structure and composition objects for featurization
- All the structures are approximated for their partial site occupancy.
- changed for Structure.from_file to cif_parser due to noisy cif (occupancy_tolerance = 2.0)
- Only the first species with occupancy >= 0.5 is kept and removed the other species. (site occupancy is more than 1!)

## Bug
-

## Future version
-
