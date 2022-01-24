#!/usr/bin/perl

die "Usage: $0 replacement-rules-file" unless -r $ARGV[0];

open(F, "$ARGV[0]");

while (my $l = <F>)
{
    next if $l =~ m/^\s*$/;
    next if $l =~ m/^\s*#/;
    chomp $l;

    my ($from, $to) = $l =~ m/^\s*(\w+)(?:\s+(\w+)\s*)?$/;
    $to = lcfirst($from) unless defined $to;

    # my @matches = `find . -name \*.h -or -name \*.cc | xargs grep $from`;
    # print "Replace '$from' --> '$to' in\n ", join(" ", @matches), "\n";

    my @matched_files = split("\n", `find . -name \*.h -or -name \*.cc | xargs grep -l $from`);

    next unless @matched_files;

    print "Replace '$from' --> '$to' in ", join(" ", @matched_files), "\n";

    for my $fname (@matched_files)
    {
        my $xxx = $/; undef $/;
        open(X, '<', $fname) or die "Can not open $fname for reading";
        my $file = <X>;
        close(X);
        $/ = $xxx;

        $file =~ s/(::|\s|\.|->|"|\(|\[)${from}(\s*\()/$1${to}$2/msg;

        my @matches = $file=~m/^.*$from.*$/mg;
        print $fname, "\n  ", join("\n  ", @matches), "\n";

        open(X, '>', $fname) or die "Can not open $fname for writing";
        print X $file;
        close(X);
    }

    print "\n";
}

close(F);
