# CS252S22
CS 252 Notebooks for Spring 2022

* Forking this git repository (get your own copy): log in to [github](https://github.com) and go to [this page](https://github.com/ajstent/CS252S22) and click "Fork"
  * In Settings:
    * Make sure you set the repository to 'Private'
    * Make sure you add (ajstent) as a Collaborator
* Using git on the commandline (including in a jupyterhub terminal):
  * Getting your fork (copy) of the repository: git clone git@github.com:(yourusername)/CS252S22.git
  * Updating your repository after you have...
    * Changed a file: 
      * git commit -m 'This is how I changed these files' .
      * git push origin main
    * Added a file
      * git add <file I added>
      * Then see "Changed a file"
    * Removed a file
      * git rm <file I want to go away>
      * Then see "Changed a file"
* Using git from VSCode: [docs](https://docs.microsoft.com/en-us/learn/modules/use-git-from-vs-code/)
* Using git from jupyterhub:
  * first, you might need to set a ssh token in github
    * from the jupyterhub home page, click New then choose Terminal
    * type ssh-keygen and follow the instructions; you don't need to supply a name or passcode
    * copy the text in .ssh/id_rsa.pub into https://github.com/settings/keys and call it colby-jupyterhub
  * then, you need to clone (checkout) the repository
    * in the terminal window, make sure you are in the folder where you want to be and type: git clone git@github.com:(yourusername)/CS252S22.git 
 
* Syncing your fork of this repository with my copy: inside the directory CS252S22,
 * Add a new remote upstream repository; you only need to do this once: git remote add upstream git@github.com:ajstent/CS252S22.git
 * sync: type these in order
   * git fetch upstream
   * git checkout main
   * git merge upstream/main
