#!/usr/bin/perl

@headers = grep { chomp; $_ !~ m!(attic|Ice|CMS-2017|MatriplexCommon|binnor)!; } `find . -name \*.h`;

# 1. Grep for classes, structs - so we don't try to fix ctors and dtors.

%ClassesStructs = ();
%Funcs = ();

%HeaderText = ();

for my $h (@headers)
{
    # local $/ = undef;
    open F, $h;
    my @lines = <F>;
    close F;

    # filter out //-style commented lines
    @lines = grep { $_ !~ m!$\s*//!o; } @lines;

    my $f = join('', @lines);

    # filter out /*-style commented code
    $f =~ s!/\*.*?\*/!!omsg;

    my @css = $f =~ m/\s(?:class|struct)\s+(\w+)/omsg;

    # print "In $h: ", join(" ", @css), "\n";

    for my $cs (@css)
    {
        $ClassesStructs{$cs} = 1;
    }

    $HeaderText{$h} = $f;
}

# 2. Grep for stuff that looks like fuctions

for my $h (keys %HeaderText)
{
    my @foos = $HeaderText{$h} =~ m/\s([A-Z]\w+)\([^)]*\)\s*(?:const)?\s*(?:;|{)/omsg;

    # print "In $h: ", join(" ", @foos), "\n";

    my @ffoos;
    for my $foo (@foos)
    {
        next if exists $ClassesStructs{$foo};
        next if exists $Funcs{$foo};
        $Funcs{$foo} = 1;

        # Needed just for printout
        push @ffoos, $foo;
    }

    if (scalar @ffoos)
    {
        print "# In $h:\n  ", join("\n  ", @ffoos), "\n";
    }
}