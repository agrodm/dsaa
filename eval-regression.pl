#!/usr/bin/perl
#
#
use warnings;
use strict;

my $resultFile = $ARGV[0];
open(RES, $resultFile);
my @resultLines = <RES>;
close(RES);
chomp(@resultLines);

my $avg = 0;
my $counter = 0;
my $all = 0;
my $correct = 0;

for(my $l = 0; $l < scalar(@resultLines); $l++){
	if($resultLines[$l] =~ /^Moda/){
		if($all != 0){
			print $correct."/".$all."\n";
			$avg += $correct/$all;
		}
		$counter++;
		$all = 0;
		$correct = 0;
	}elsif($resultLines[$l] =~ /sample/){
		$resultLines[$l] =~ s/(\{)|(\})|(')|( )//g;
		my @parts = split(/:/, $resultLines[$l]);
		print $parts[-1]." : ".$resultLines[$l+1]."\n";
		$all++;
		if($parts[-1] eq $resultLines[$l+1]){
			$correct++;
		}else{
			print "mismatch\n";
		}
	}
}

if($all != 0){
	print $correct."/".$all."\n";
	$avg += $correct/$all;
}

print "".($avg/$counter)."\n";
