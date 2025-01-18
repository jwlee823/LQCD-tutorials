#!/usr/local/bin/perl

$a = 1;

until ( $a > 200 )
{
   $b = 20*$a+1000;
#   print "/home/user1/jwlee823/code/HiRep_SVN_MR_v2/HMC/configs_24x12x12x12b6.4mas-1.035/run1_24x12x12x12nc4rASYnf5b6.400000m1.035000n$b\n";
#   print "/home/user1/jwlee823/code/HiRep_SVN/PureGauge/configs_60x48x48x48b8.0/run1_60x48x48x48nc4b8.000000n$b\n";
#   print "/data/jwlee823/configs_SPN/configs_48x24x24x24b7.5m0.7/run1_48x24x24x24nc4rASYnf2b7.500000m0.700000n$b\n";
   print "/Volumes/data/physics/SPN/code/HiRep_SVN_MR_v2/HMC/b6.7/run1_4x4x4x4nc4rASYnf5b6.700000m0.800000n$b\n";
   $a = $a + 1;
}
