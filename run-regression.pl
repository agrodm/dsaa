#!/usr/bin/perl
#
#
use warnings;
use strict;

my $ds = $ARGV[0];
my $output = $ARGV[1];
my $repetitions = $ARGV[2];

for(my $r = 0; $r < $repetitions; $r++){
	my $cmd = "python3 regressionpls.py ".$ds." >> ".$output;
	print "Iteration ".$r.": $cmd\n";
	`$cmd`;
}
