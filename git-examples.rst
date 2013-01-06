Examples of git usage
=====================

Read http://sitaramc.github.com/gcs/index.html for definition of some git terms.

Apply the configuration described in HOWTO-dev.rst.


Clone the remote repository:: 

  git clone USER@bumps.ugent.be:/srv/git/pyformex.git

This creates a directory `pyformex`, with a clone of the remote repository
in `pyformex/.git` and a checked out working directory in `pyformex`::

  cd pyformex
  ls
  ls -a .git

.. note: Difference with Subversion: Each user has a full clone of the 
   repository. Checking in (commit) will happen first with respect to 
   your own copy. Afterwards, you can push commits to the remote repository.

   Another difference is that git only creates a *.git* hidden subdirecotory
   in the top checkout path, while Subversion created *.svn* subdirectories
   on all lower levels. 

   If you only want to checkout a repository to run the
   source, and have no intention to do any development nor update in this path, 
   you can safely remove *.git*

You can *fetch* from and *push* to multiple remote repositories. You can see which ones are currently configured with ::

  git remote -v

.. note: Until we have establish more complex workflows, we suggest you
   keep your remotes limited to `USER@bumps.ugent.be`.

A repository may contain multiple branches. To show them all (local and remote)::

   git br -a

Branches will become important in future. The old branches in the repository
will be removed. We may introduce a 'development' branch. There will 
certainly be a branching when reaching version 1.0. Another branch will
be introduced for the new OpenGl rendering engine in pyFormex. Your local
repository may follow any branch, but for now, we will limit us to the 
default *master* branch.

You can also create local branches, and you are encouraged to do so. By creating
local branches, you can easily do some partial work, switch to another branch to
do some other work (e.g. fixing a bug), commit that work, and then go back to
your first branch to continue that work (which will not be included in the commit).

Another reason to create a local branch, is if you want to add code that you do
not want to share (yet), or want to share only with a limited number of users. 
This code should then not be pushed to the public repository.

We will deal with the use of branches later, let's first do some work in the master branch.  

A branch can also contain a number of tags, pointing to a particular interesting commit. Currently we have a tag for each released version::

  git tag

produces ::

  release-0.2
  release-0.2.2
  release-0.3
  release-0.3.1-alpha
  release-0.4
  release-0.4.1
  release-0.4.2
  release-0.5
  release-0.6
  release-0.7
  release-0.7.1
  release-0.7.2
  release-0.7.3
  release-0.8
  release-0.8.1
  release-0.8.2
  release-0.8.3
  release-0.8.4
  release-0.8.5
  release-0.8.6
  release-0.8.8
  release-0.8.9

These can be used to go back to a certain commit of the past. You can create
as many tags as you want on you local branches, but you should normally not
push up your tags to the repository: another user may want to use his own set
 of tags. The release manager however will push up the tags created to point to 
official releases. Occasionally, tags for other important commits may get 
pushed. 

Of course you can also restore the situation of any other past commit (those
 without a tag). But then you have to use the corresponding SHA hash.

.. note: Difference with Subversion: in subversion, all commits are done to 
   a signle repository and are numbered consecutively (the revision number).
   In git, there is no such number, because there is no single repository.

In git, individual commits are identified by a unique SHA hash number. 
The following commands show the last commits::

  git log

The result shows the commit SHA, author and date info, and commit message::

  commit 3195eafac4759bd076fc0d0451011ea380043630
  Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>
  Date:   Fri Jan 4 12:29:38 2013 +0100

      Changed source checkout for migration to git

  commit 8d8d85394476a755d6f7de389e6052337d22a88d
  Merge: 2e6f4c9 6c8d710
  Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>
  Date:   Fri Jan 4 09:30:43 2013 +0100

      Merge branch 'master' of git.sv.gnu.org:/srv/git/pyformex

  commit 2e6f4c98de26754f44944108bdee5e695627b235
  Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>
  Date:   Fri Jan 4 09:30:16 2013 +0100

      Added gitref.org to git doc links

In git commands, you can use an abbreviated SHA number to identify the commit.
Look through the history above for examples.

OK, let's do some work. First make sure we are on a clear master branch::

  git st

yields::

  # On branch master
  nothing to commit (working directory clean)

Now make some changes, like editing a file and/or adding a new file. After that do::

 git status

You get something like::

  # On branch master
  # Changes not staged for commit:
  #   (use "git add <file>..." to update what will be committed)
  #   (use "git checkout -- <file>..." to discard changes in working directory)
  #
  #	modified:   HOWTO-dev.rst
  #
  # Untracked files:
  #   (use "git add <file>..." to include in what will be committed)
  #
  #	git-examples.rst
  no changes added to commit (use "git add" and/or "git commit -a")

This shows you have modified `HOWTO-dev.rst` (which is already tracked) and
you have a new untracked file `git-examples.rst`. Suppose you want to get these
changes in the repo (my local one!). First you should add the changes::

  git add HOWTO-dev.rst 
  git add git-examples.rst
  git status

  # On branch master
  # Changes to be committed:
  #   (use "git reset HEAD <file>..." to unstage)
  #
  #	modified:   HOWTO-dev.rst
  #	new file:   git-examples.rst
  #

Now the changes are ready to be committed to the repo::

  git commit

Like in subversion, an editor will show up where you should enter a commit message. We recommend (maybe we should enforce?) to enter detailed commit messages, consisting of a single short (max 50 chars) header line, a blank line and 
multiple detail lines (by preference not longer than 72 characters). ::

  Added to developer documentation

  Added new file git-examples.rst, with an overview of git usage for 
  pyFormex.
  Unimportant change in HOWTO-dev.rst

If you leave an empty message, the commit will be aborted. After a succesful commit the status looks like::

  # On branch master
  # Your branch is ahead of 'origin/master' by 1 commit.
  #
  nothing to commit (working directory clean)

We again have a clean working directory, ready for more work. Remark that if
you want to check in *all* your changes, you can do the *add* and *commit* in
a single command::

  git commit -a

This only works for files that are already tracked. New files always need to
be added first. 

Remember that all the commits that you make, are only to your local copy of the
repository. This can also be seen from the status command::

  # On branch master
  # Your branch is ahead of 'origin/master' by 3 commits.
  #
  nothing to commit (working directory clean)

If you want to push the changes to the remote repository, do ::

  git push


Working with multiple branches
------------------------------

.. note: This needs to be added

Working with multiple repos
---------------------------

Add another remote repo::

  git remote add public	bverheg@git.sv.gnu.org:/srv/git/pyformex.git

Now the command ::

  git remote -v

gives::

  origin	bene@bumps.ugent.be:/srv/git/pyformex.git (fetch)
  origin	bene@bumps.ugent.be:/srv/git/pyformex.git (push)
  public	bverheg@git.sv.gnu.org:/srv/git/pyformex.git (fetch)
  public	bverheg@git.sv.gnu.org:/srv/git/pyformex.git (push)

The default used is origin (the one I cloned from). The public is where I
push changes to make them available to the general public.


.. End
