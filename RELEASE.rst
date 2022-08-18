How to release a new version of glue-astronomy
==============================================

#. Edit the ``CHANGES.rst`` file to add the release date for the release
   you want to make and make sure the changelog is complete.

#. Commit the changes using::

    git commit -m "Preparing release v..."

   where v... is the version you are releasing and push to main::

    git push upstream main

#. Tag the release you want to make, optionally signing it (``-s``)::

    git tag -m v0.5.0 v0.5.0

   and push the tag::

    git push upstream v0.5.0

#. At this point, the release sdist and wheel will be built on by GitHub
   Actions and automatically uploaded to PyPI. You can check the build
   for the release commit `here <https://github.com/glue-viz/glue-astronomy/actions/>`_
   and if there are any issues you can delete the tag, fix the issues
   (preferably via a pull request) and then try the release process
   again.
