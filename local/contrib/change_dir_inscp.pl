#!/usr/bin/env perl
#===============================================================================
#
#         FILE: change_dir_inscp.pl
#
#        USAGE: ./change_dir_inscp.pl  
#
#  DESCRIPTION: change directory root in given scp file 
#
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
#       AUTHOR: YOUR NAME (), 
# ORGANIZATION: 
#      VERSION: 1.0
#      CREATED: 09/26/2019 07:17:04 AM
#     REVISION: ---
#===============================================================================

use strict;
use warnings;
use utf8;
my $ori_dir = $ARGV[0];
my $new_dir = $ARGV[1];
print $ori_dir."\n==>".$new_dir."\nConverting...\n";
my $arg_num = @ARGV;
my @scp_files = @ARGV[2..$arg_num-1];

foreach my $scp (@scp_files) {
	chomp $scp;
	print $scp."\n";
	my $tmp_scp = $scp.".tmp";
	`mv $scp $tmp_scp`;
	my $no_new_line = 1;
	open(SCP,"<","$tmp_scp") or die $!;
	open(NEW,">","$scp") or die $!;
	foreach my $line(<SCP>){
		chomp $line;
		$line =~ s/ $ori_dir/ $new_dir/;
		if($no_new_line==1){
			print NEW $line;
			$no_new_line = 0;
		} else {
			print NEW "\n".$line;
		}
	}
	close(SCP);
	close(NEW);
	`rm $tmp_scp`;
}
